# a collection of region segmentation and sorting routines that use the magnetic squashing factor

# general packages
import numpy as np
import scipy as sp
import matplotlib as mpl
from matplotlib.widgets import Slider, RadioButtons, Button
import pylab as plt
import pandas as pd
import skimage as ski
from skimage import morphology as mor
import skimage.measure as skim
from scipy import ndimage as ndi
from scipy.interpolate import RegularGridInterpolator as RGI


#### optional printing
#    import logging
#    print = logging.info
#    logging.basicConfig(level=logging.WARNING if QUIET else logging.INFO, format="%(message)s")

import time
import os
import pdb
    
########################################################################
# grid extent and size must match corresponding definitions in options.hpp and snapshot.cpp
########################################################################


def gauss_slice(cube, axis, center=0.5, sigma=0.1):
    nx,ny,nz = cube.shape
    tmpx = np.linspace(0, 1, nx)
    tmpy = np.linspace(0, 1, ny)
    tmpz = np.linspace(0, 1, nz)
    if axis==0: tmp = np.reshape(np.roll(np.exp(- 0.5 * ((tmpx-0.5) / sigma)**2), int((center - 0.5)*nx) ), [tmpx.size,1,1])
    if axis==1: tmp = np.reshape(np.roll(np.exp(- 0.5 * ((tmpy-0.5) / sigma)**2), int((center - 0.5)*ny) ), [1,tmpy.size,1])
    if axis==3: tmp = np.reshape(np.roll(np.exp(- 0.5 * ((tmpz-0.5) / sigma)**2), int((center - 0.5)*nz) ), [1,1,tmpz.size])
    return np.mean(tmp*cube, axis=axis)/np.mean(tmp, axis=axis)
    
    
def set_grid(nr=None, nz_out=None, ny_out=None, nx_out=None, xmin=None, xmax=None, ymin=None, ymax=None, zmin=None, zmax=None, z_samp=None, solrad=None):
    if nr==None: nr=150
    if nx_out==None: nx_out = nr * 8
    if ny_out==None: ny_out = nr * 4
    if nz_out==None: nz_out = nr
    if xmin==None: xmin   =  0.
    if xmax==None: xmax   =  360.   
    if ymin==None: ymin   = -88.   
    if ymax==None: ymax   =  88.   
    if zmin==None: zmin   =  0.0
    if zmax==None: zmax   =  1044. # z in height above solar surface
    if solrad==None: solrad = 696.
    if z_samp==None: z_samp='log'

    from numpy import exp, log

    def z_sampler(z1): # must be consistent with other routines
        z=float(z1)
        if z_samp=='linear':
            z=z*(zmax-zmin)+zmin
        if z_samp=='svet_exp':
            z=(exp(((70.+z*300.)/370.)**3)-exp((70./370.)**3))/(exp(1.)-exp((70./370.)**3))*(zmax-zmin)+zmin
        if z_samp=='log':
            z = (solrad+zmax)**z * (solrad+zmin)**(1-z) - solrad
        return z

    Lx      = xmax-xmin
    Ly      = ymax-ymin

    xx=((np.array(range(nx_out),dtype='float32'))/(nx_out-1.0)*Lx+xmin)
    yy=((np.array(range(ny_out),dtype='float32'))/(ny_out-1.0)*Ly+ymin)

    z_sampler=np.vectorize(z_sampler)
    zz=z_sampler(np.array(range(nz_out),dtype='float32')/(nz_out-1))

    return nx_out, ny_out, nz_out, xx, yy, zz, solrad











    

def extract_squash_data(data=None, nr=None, nz_out=None, ny_out=None, nx_out=None, xmin=None, xmax=None, ymin=None, ymax=None, zmin=None, zmax=None, z_samp=None, solrad=None, q_dir=None, ss_eof=None):

    if q_dir==None: q_dir='./'
    if data==None: data={}
    nx_out, ny_out, nz_out, xx, yy, zz, solrad = set_grid(nr=nr, nz_out=nz_out, ny_out=ny_out, nx_out=nx_out, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, zmin=zmin, zmax=zmax, z_samp=z_samp, solrad=solrad)
    
    print('Loading Q data \n%%%%%%%%%%%%%%%%%%%%%%%%%%%%')

    ########################################################################
    # Import q in spherical coordinates:
    ########################################################################

    qarr=pd.read_table(q_dir+'/grid3d.dat',header=None).astype('float32')
    slog10q=(np.array(qarr))[...,0].reshape((nx_out,ny_out,nz_out))
    del qarr
            
    if (ss_eof==True): slog10q[...,-1] = -np.absolute(slog10q[...,-1])

    crr=np.repeat(np.repeat([[zz]],ny_out,axis=1),nx_out,axis=0) + solrad
    cph=np.repeat(np.repeat([[xx]],ny_out,axis=1),nz_out,axis=0).transpose((2,1,0))
    cth=np.repeat(np.repeat([[yy]],nx_out,axis=1),nz_out,axis=0).transpose((1,2,0))

    data['crr'] = crr.astype('float32')
    data['cth'] = cth.astype('float32')
    data['cph'] = cph.astype('float32')
    data['slog10q'] = slog10q
    data['solrad'] = solrad
    data['q_dir']=q_dir
    return data


def get_differentials(data=None, nr=None, nz_out=None, ny_out=None, nx_out=None, xmin=None, xmax=None, ymin=None, ymax=None, zmin=None, zmax=None, z_samp=None, solrad=None, q_dir=None, b_dir=None, glbl_mdl=None, ss_eof=None):

    # must have data cube before continuing
    
    if (data==None or ('q_dir' not in data.keys() ) ):
        data=extract_squash_data(data=data, nr=nr, nz_out=nz_out, ny_out=ny_out, nx_out=nx_out, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, zmin=zmin, zmax=zmax, z_samp=z_samp, solrad=solrad, q_dir=q_dir, ss_eof=ss_eof)

    drr         = np.gradient(data['crr'], axis=2)
    rr_dth      = np.gradient(data['cth'], axis=1) * np.pi/180 * data['crr']
    rr_csth_dph = np.gradient(data['cph'], axis=0) * np.pi/180 * data['crr'] * np.cos(data['cth'] * np.pi/180)

    # naming convention here is a bit weird but these are just dr, r dth, r sinth dph
    return rr_csth_dph, rr_dth, drr

    
def extract_bfield_data(data=None, nr=None, nz_out=None, ny_out=None, nx_out=None, xmin=None, xmax=None, ymin=None, ymax=None, zmin=None, zmax=None, z_samp=None, solrad=None, q_dir=None, b_dir=None, glbl_mdl=None, ss_eof=None):
    
    if glbl_mdl==None: glbl_mdl=True

    # must have squash data before extracting bfield
    if (data==None or ('q_dir' not in data.keys() ) ):
        data=extract_squash_data(data=data, nr=nr, nz_out=nz_out, ny_out=ny_out, nx_out=nx_out, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, zmin=zmin, zmax=zmax, z_samp=z_samp, solrad=solrad, q_dir=q_dir, ss_eof=ss_eof)
    
    # default directory layout
    if b_dir==None:
        b_dir=data['q_dir']+'/bfield_data'

    solrad=data['solrad']
        
    print('Loading B field data \n%%%%%%%%%%%%%%%%%%%%%%%%%%%%')

    ########################################################################
    # Import B field in spherical coordinates:
    ########################################################################

    ph_mag = np.array(pd.read_table(b_dir+'/xs0.dat',header=None))[...,0].astype('float32')
    th_mag = np.array(pd.read_table(b_dir+'/ys0.dat',header=None))[...,0].astype('float32')
    rr_mag = np.array(pd.read_table(b_dir+'/zs0.dat',header=None))[...,0].astype('float32') * solrad

    nx = ph_mag.size
    ny = th_mag.size
    nz = rr_mag.size

    #minus in b_theta is due to Griffiths' def
    b_ph_nat =   (np.array(pd.read_table(b_dir+'/bx0.dat',header=None))[...,0].reshape((nz,ny,nx)) ).transpose((2,1,0)).astype('float32')
    b_th_nat = - (np.array(pd.read_table(b_dir+'/by0.dat',header=None))[...,0].reshape((nz,ny,nx)) ).transpose((2,1,0)).astype('float32')
    b_rr_nat =   (np.array(pd.read_table(b_dir+'/bz0.dat',header=None))[...,0].reshape((nz,ny,nx)) ).transpose((2,1,0)).astype('float32')
    if np.sum(np.array((np.isnan(b_ph_nat), np.isnan(b_th_nat), np.isnan(b_rr_nat)))) > 0:
        if np.sum(np.isnan(b_ph_nat)) > 0:
            print('Nan found in Bph')
        if np.sum(np.isnan(b_th_nat)) > 0:
            print('Nan found in Bth')
        if np.sum(np.isnan(b_rr_nat)) > 0:
            print('Nan found in Brr')
        return

    ###### global models exclude ph=360 from the source field but include it in the q output.
    ###### global models also exclude the poles. We reintroduce these, setting the pole equal to the ave.
    if glbl_mdl==True:
        nx += 1
        ph_mag = np.append(ph_mag, 360.)

        b_ph_nat = np.append(b_ph_nat, b_ph_nat[0,:,:].reshape(1,ny,nz), axis=0)
        b_th_nat = np.append(b_th_nat, b_th_nat[0,:,:].reshape(1,ny,nz), axis=0)
        b_rr_nat = np.append(b_rr_nat, b_rr_nat[0,:,:].reshape(1,ny,nz), axis=0)
        ##### b_ph is now explicitly periodic and spans the entire domain.

        ny += 2
        th_mag = np.append(np.append(-90, th_mag), 90)
        
        ### set the value at the poles equal to the mean of the next cell row. 
        b_ph_nat = np.append(np.append(np.mean(b_ph_nat[:,0,:], axis=0)*np.ones((nx,1,1)), b_ph_nat, axis=1), np.mean(b_ph_nat[:,-1,:], axis=0)*np.ones((nx,1,1)), axis=1)
        b_th_nat = np.append(np.append(np.mean(b_th_nat[:,0,:], axis=0)*np.ones((nx,1,1)), b_th_nat, axis=1), np.mean(b_th_nat[:,-1,:], axis=0)*np.ones((nx,1,1)), axis=1)
        b_rr_nat = np.append(np.append(np.mean(b_rr_nat[:,0,:], axis=0)*np.ones((nx,1,1)), b_rr_nat, axis=1), np.mean(b_rr_nat[:,-1,:], axis=0)*np.ones((nx,1,1)), axis=1)

    print('Interpolating B field onto Q grid \n%%%%%%%%%%%%%%%%%%%%%%%%%%%%')

    b_coords = [ph_mag, th_mag, rr_mag]
    q_pts = np.stack(( data['cph'], data['cth'], data['crr'] )).reshape((3, data['cph'].size)).T

    bph_interpolator = RGI(points=b_coords, values=b_ph_nat)
    bth_interpolator = RGI(points=b_coords, values=b_th_nat)
    brr_interpolator = RGI(points=b_coords, values=b_rr_nat)   
 
    bph = bph_interpolator(q_pts).T.reshape(data['cph'].shape)
    bth = bth_interpolator(q_pts).T.reshape(data['cph'].shape)
    brr = brr_interpolator(q_pts).T.reshape(data['cph'].shape)
            
    data['brr'] = brr.astype('float32')
    data['bth'] = bth.astype('float32')
    data['bph'] = bph.astype('float32')
    data['b_dir'] = b_dir

    null_loc_fname = b_dir+'/nullpositions.dat'
    if os.path.isfile(null_loc_fname):
        from scipy.io import readsav
        null_dict=readsav(null_loc_fname, python_dict=True)
        null_locs=null_dict['nulls']
        data['null_locs']=null_locs
    
    return data




    

        

def build_qsl_msk(data=None, nr=None, nz_out=None, ny_out=None, nx_out=None, xmin=None, xmax=None, ymin=None, ymax=None, zmin=None, zmax=None, z_samp=None, solrad=None, q_dir=None, b_dir=None, glbl_mdl=None, ss_eof=None, nrm_thrsh=None, save_sobel=None):

    if (data == None):
        data=extract_bfield_data(data=data, nr=nr, nz_out=nz_out, ny_out=ny_out, nx_out=nx_out, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, zmin=zmin, zmax=zmax, z_samp=z_samp, solrad=solrad, q_dir=q_dir, b_dir=b_dir, glbl_mdl=glbl_mdl, ss_eof=ss_eof)

    if ('brr' not in data.keys()):
        data=extract_bfield_data(data=data, nr=nr, nz_out=nz_out, ny_out=ny_out, nx_out=nx_out, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, zmin=zmin, zmax=zmax, z_samp=z_samp, solrad=solrad, q_dir=q_dir, b_dir=b_dir, glbl_mdl=glbl_mdl, ss_eof=ss_eof)


    #######################################
    # here we start our feature detection
    #######################################

    if nrm_thrsh==None: nrm_thrsh=4.5
    slog10q = data['slog10q']
        
    ### get the derivs.
    ### normalize derivs. want r*grad so deriv scales to radius
    ### in spherical, grad is d/dr, 1/r d/dth, 1/rsinth d/dph
    dslq_dph, dslq_dth, dslq_drr = np.gradient(np.log(10.) * slog10q, np.pi*data['cph'][:,0,0]/180, np.pi*data['cth'][0,:,0]/180, data['crr'][0,0,:], axis=(0,1,2))
    gdn_slq_ph = dslq_dph / np.cos(np.pi * data['cth'] / 180)
    del dslq_dph
    gdn_slq_th = dslq_dth
    del dslq_dth
    gdn_slq_rr = dslq_drr * data['crr']
    del dslq_drr
    absQp = 10.**np.clip(np.absolute(slog10q), np.log10(2), 10)
    
    # ### construct grad Q magnitude
    GlnQp = np.sqrt(gdn_slq_ph**2 + gdn_slq_th**2 + gdn_slq_rr**2)
    #del gdn_slq_ph, gdn_slq_th, gdn_slq_rr, abs_sqp
    print('Gradient norm constructed \n%%%%%%%%%%%%%%%%%%%%%%%%%%%%')

    
    ### create mask where norm greater than thresh
    ### empirically it seems that GlnQp goes as r^(-1/2) in the volume, so we'll incorporate this into the mask.
    
    SBN_val = absQp + (data['crr'] / data['crr'].min()) * GlnQp**2 ## sets threshold at outer radius and then scales inward.
    qsl_msk = np.log10(SBN_val) > nrm_thrsh    
    
    print('Threshold / edge mask built \n%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
    ### filter using opening closing / object/hole removal
    qsl_msk = mor.closing(qsl_msk)
    print('Mask holes removed \n%%%%%%%%%%%%%%%%%%%%%%%%%%%%')

    data['qsl_msk'] = qsl_msk
    data['GlnQp'] = np.clip(GlnQp, 0, 10**10).astype('float32')
    
    return data








def segment_volume(data=None, nr=None, nz_out=None, ny_out=None, nx_out=None, xmin=None, xmax=None, ymin=None, ymax=None, zmin=None, zmax=None, z_samp=None, solrad=None, q_dir=None, b_dir=None, glbl_mdl=None, ss_eof=None, nrm_thrsh=None, pad_ratio=None, bot_rad=None):
    
    if (data==None or ('qsl_msk' not in data.keys() ) ):
        data=build_qsl_msk(data=data, nr=nr, nz_out=nz_out, ny_out=ny_out, nx_out=nx_out, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, zmin=zmin, zmax=zmax, z_samp=z_samp, solrad=solrad, q_dir=q_dir, b_dir=b_dir, glbl_mdl=glbl_mdl, ss_eof=ss_eof, nrm_thrsh=nrm_thrsh)

    # now we'll set up our regions for labeling and growing
    # first we do a simple discrete labeling, with a conservative mask
    # then we use region growing to backfill through the mask.

    print('Beginning volume segmentation \n%%%%%%%%%%%%%%%%%%%%%%%%%')
    
    solrad=data['crr'][0,0,0]
    if bot_rad==None: bot_rad=1.0
    
    slog10q = data['slog10q']
    qsl_msk = data['qsl_msk']

    nph, nth, nrr = qsl_msk.shape
    
    q_segment = np.zeros(slog10q.shape, dtype='int32')### initiate segmentation label array.
    # it's useful to have masks for open and closed flux. We consider undecided flux to be open.
    opn_msk = (slog10q < 0)
    if ss_eof==True: opn_msk[...,-1]=True
    cls_msk = (slog10q > 0)
    nll_msk = (~opn_msk) & (~cls_msk)

    print('Calculating distance transforms \n%%%%%%%%%%%%%%%%%%%%%%%%%')

    # it's also useful to have the distance from the interior of the qsl regions to their boundary, and the other way.
    qsl_dist = ndi.distance_transform_edt(~qsl_msk).astype('float32') # distance to nearest qsl
    reg_width = np.mean(qsl_dist[np.nonzero(~qsl_msk)])*4 # full width is 4 times average distance to boundary
    print('Vol width: ',reg_width)
    reg_dist = ndi.distance_transform_edt( qsl_msk).astype('float32') # distance to nearest low q region
    qsl_width = np.mean(reg_dist[np.nonzero( qsl_msk)])*4 # full width is 4 times average distance to boundary
    print('HQV width: ',qsl_width)

    
    print('Performing discrete flux labeling \n%%%%%%%%%%%%%%%%%%%%%%%%%')

    if pad_ratio==None: pad_ratio=0.25 # default is pad out to a typical quarter-width.
    
    # now we'll label the open flux domains, above min height.
    # we'll pad the qsl mask by a distance proportionate to the qsl_halfwidth, including a radial scaling to accomodate thicker hqvs at larger radii
    pad_msk = ( qsl_dist > pad_ratio * qsl_width * (data['crr'] / data['crr'].mean()) ) # empirical radial dependence...
    q_segment -= skim.label(pad_msk & (slog10q < 0) & (data['crr'] >= bot_rad*solrad)) # all pixels not within or adjacent to a qsl
    open_labels = np.unique(q_segment[np.nonzero(q_segment < 0)])
    # and we'll get the closed flux labels in the same way, also above min height.
    q_segment += skim.label(pad_msk & (slog10q > 0) & (data['crr'] >= bot_rad*solrad)) # all pixels not within or adjacent to a qsl
    clsd_labels = np.unique(q_segment[np.nonzero(q_segment > 0)])
    data['pad_msk'] = pad_msk
    del slog10q, pad_msk

    print('Associating domains across ph=0 boundary \n%%%%%%%%%%%%%%%%%%%%%%%%%')

    phi_bound_thresh=0.5
    
    # now we'll associate regions across the phi boundary
    # need the labels on the boundary
    phb_seg = np.roll(q_segment, 1, axis=0)[0:2,...] # this is just the 0,-1 columns stacked together
    phb_regs = np.unique(phb_seg[np.nonzero(phb_seg)]) # these are the regions associated with the boundary
    # and we need to know if the open regions are positive or negative to avoid accidental mixing.
    opnflxpos=(phb_regs!=phb_regs) # this will boolean to say whether each region is [open and positive]
    for i in np.arange(phb_regs.size):
        ri = phb_regs[i]
        if ri < 0: # test if open
            n_pos = np.sum(data['brr'][...,-1][np.nonzero(q_segment[...,-1]==ri)]>0) # positive area
            n_neg = np.sum(data['brr'][...,-1][np.nonzero(q_segment[...,-1]==ri)]<0) # negative area
            opnflxpos[i] = (n_pos > n_neg) # true if pos > neg -- this list is populated at least as fast as it is queried.

    # now we can create a table of areas to be swapped
    nregs = phb_regs.size
    swap=np.zeros((nregs,nregs)).astype('bool')
    for i in np.arange(nregs):
        ri = phb_regs[i]
        iopos=opnflxpos[i]
        for j in np.arange(nregs):
            rj = phb_regs[j]
            jopos=opnflxpos[j]
            # we'll do a double mask and multiply.
            # if over threshold fit, then there are more matches on that pairing than all others combined.
            if ((ri != rj) & (ri*rj > 0) & (iopos==jopos)): # only consider same type flux -- skip redundant names.
                mi = ((phb_seg[0,...]==ri) & (qsl_msk[0,...]==0))
                mj = ((phb_seg[1,...]==rj) & (qsl_msk[1,...]==0))
                i_area = np.sum( mi ) 
                j_area = np.sum( mj )
                c_area = np.sum( (mi & mj) )
                if (c_area > 0):
                    # compare c_area to min allowable: composed of the larger of ( local weighted qsl cross section , smaller of ( fraction of local individual areas))
                    qsl_local_area_ave = np.sum( ( np.pi * (0.5*qsl_width)**2 * data['crr'][0,...]**2 / data['crr'].mean()**2 ) * (mi & mj) ) / c_area
                    if c_area > np.max(( qsl_local_area_ave, phi_bound_thresh * np.min(( i_area, j_area )) )):        
                        swap[i,j]=True

    # now the actual swapping
    swap = swap | swap.T     # first we consolidate the swap map to the minor diagonal
    for i in np.arange(nregs):
        ri = phb_regs[i]
        i_swapped=False # this region hasn't been updated yet
        for j in np.arange(i+1, nregs): # previous entries are redundant.
            rj = phb_regs[j]
            if (i_swapped==False):
                if swap[i,j]==True:
                    q_segment[np.nonzero(q_segment==ri)]=rj # i label -> j label, which hasn't be processed
                    swap[j,:]=swap[i,:] # j entries inherit swapability from i entries. 
                    i_swapped=True # ri has been swapped up to rj

    print('Removing domains with sub-minimum volume \n%%%%%%%%%%%%%%%%%%%%%%%%%')

    for reg in np.unique(q_segment[np.nonzero(q_segment)]):
        tmp_msk = (q_segment == reg)
        #print('region: '+str(reg)+', volume: '+str(np.sum(tmp_msk)))
        if np.sum(tmp_msk * data['crr'].mean()**2 / data['crr']**2) < (0.5*qsl_width)**3: # threshold size for valid regions -- threshold increases quadratically with height to allow smaller closed domains
            q_segment = q_segment * ~tmp_msk  # zero in mask, unchanged else.
    del tmp_msk
    
    print('Performing watershed backfill into HQV padding \n%%%%%%%%%%%%%%%%%%%%%%%%%')

    absQp = np.clip(np.absolute(data['slog10q']), np.log10(2), 10)
    log10_SNQ =  np.log10(absQp + (data['crr'] / data['crr'].max()) * data['GlnQp']**2)
    del absQp
    
    # Initial watershed phase
    # First we grow regions into the padding layer outside the hqv mask
    stime = time.time()
    q_segment += mor.watershed(           log10_SNQ, q_segment * opn_msk, mask=(opn_msk & ~qsl_msk), watershed_line=False) * ((q_segment==0) & opn_msk & ~qsl_msk)
    print('Open flux backfill completed in '+str(int(time.time()-stime))+' seconds')
    stime = time.time()
    q_segment += mor.watershed(           log10_SNQ, q_segment * cls_msk, mask=(cls_msk & ~qsl_msk), watershed_line=False) * ((q_segment==0) & cls_msk & ~qsl_msk)
    print('Clsd flux backfill completed in '+str(int(time.time()-stime))+' seconds \n%%%%%%%%%%%%%%%%%%%%%%%%%')
    
    
    print('Enforcing boundary connectivity \n%%%%%%%%%%%%%%%%%%%%%%%%%')

    # And we require that all regions are associated with a boundary
    # These loops are split up to allow for different definitions between open and closed domains
    for reg in open_labels:
        tmp_msk = (q_segment == reg)
        if np.sum(tmp_msk[...,-1]) == 0: # open domains must intersect the top boundary
            q_segment = q_segment * ~tmp_msk # zero in mask, unchanged else.
    for reg in clsd_labels:
        tmp_msk = (q_segment == reg)
        if np.sum(tmp_msk[...,0]) == 0: # closed domains must intersection the bottom boundary.
            q_segment = q_segment * ~tmp_msk # zero in mask, unchanged else.
    del tmp_msk


    
    print('Performing restricted watershed backfill into HQV mask \n%%%%%%%%%%%%%%%%%%%%%%%%%')

    # Second, we grow regions using the same-type mask, which just expands open-open, close-close.
    stime = time.time()
    q_segment += mor.watershed(           log10_SNQ, q_segment * opn_msk, mask=opn_msk, watershed_line=False) * ((q_segment==0) & opn_msk)
    print('Open flux backfill completed in '+str(int(time.time()-stime))+' seconds')
    stime = time.time()
    q_segment += mor.watershed(           log10_SNQ, q_segment * cls_msk, mask=cls_msk, watershed_line=False) * ((q_segment==0) & cls_msk)
    print('Clsd flux backfill completed in '+str(int(time.time()-stime))+' seconds \n%%%%%%%%%%%%%%%%%%%%%%%%%')


    
    print('Performing transparent watershed backfill into HQV mask \n%%%%%%%%%%%%%%%%%%%%%%%%%')

    # Third, we grow regions through opposite type, but only within a qsl, where type mixing is expected.
    stime = time.time()
    q_segment += mor.watershed(           log10_SNQ, q_segment * opn_msk, mask=qsl_msk, watershed_line=False) * ((q_segment==0) & opn_msk)
    print('Open flux backfill completed in '+str(int(time.time()-stime))+' seconds')
    stime = time.time()
    q_segment += mor.watershed(           log10_SNQ, q_segment * cls_msk, mask=qsl_msk, watershed_line=False) * ((q_segment==0) & cls_msk)
    print('Clsd flux backfill completed in '+str(int(time.time()-stime))+' seconds')


    
    print('Performing watershed backfill into residual domains \n%%%%%%%%%%%%%%%%%%%%%%%%%')
    # Finally, we grow null regions with no preference, allowing open and closed to compete.
    stime = time.time()
    q_segment += mor.watershed(           1/(1 + qsl_dist), q_segment,                         watershed_line=False) * ((q_segment==0))
    print('Final flux backfill completed in '+str(int(time.time()-stime))+' seconds \n%%%%%%%%%%%%%%%%%%%%%%%%%')
    # There may still be unnasigned regions. But these will be outliers buried deep within opposite flux types.


    
    print('Relabeling to remove obsolete domains \n%%%%%%%%%%%%%%%%%%%%%%%%%') 
    # now let's relabel with integer labels removing gaps
    open_labels = -np.unique(-q_segment[np.nonzero(q_segment < 0)]) # negative gets it in reverse order
    clsd_labels = np.unique(q_segment[np.nonzero(q_segment > 0)])
    # we want this to be a random reordering so we have to keep track of swapped regions to avoid multiple swaps.
    open_relabel = np.arange(0, open_labels.size)
    clsd_relabel = np.arange(0, clsd_labels.size)
    np.random.seed(open_relabel.size) # repeatable random seed
    np.random.shuffle(open_relabel) # random shuffle of domain order
    np.random.seed(clsd_relabel.size) # repeatable random seed
    np.random.shuffle(clsd_relabel) # random shuffle of domain order

    swapped = (q_segment != q_segment) # boolean to track already swapped domains
    for i in range(open_relabel.size):
        swap_msk = ((q_segment == open_labels[i]) & ~swapped)
        q_segment = q_segment * (~swap_msk) - (open_relabel[i]+1) * (swap_msk)
        swapped = swapped | swap_msk
    for i in range(clsd_relabel.size):
        swap_msk = ((q_segment == clsd_labels[i]) & ~swapped)
        q_segment = q_segment * (~swap_msk) + (clsd_relabel[i]+1) * (swap_msk)
        swapped = swapped | swap_msk
    del swapped, swap_msk, open_relabel, clsd_relabel
        
    open_labels = np.unique(q_segment[np.nonzero(q_segment < 0)])
    clsd_labels = np.unique(q_segment[np.nonzero(q_segment > 0)])
    labels = np.unique(q_segment[np.nonzero(q_segment)])
    
    print('Finished segmenting volume \n%%%%%%%%%%%%%%%%%%%%%%%%%')

    # and to recover our domain boundaries
    seg_gx, seg_gy, seg_gz = np.gradient(q_segment, axis=(0,1,2))
    seg_msk = ((seg_gx**2 + seg_gy**2 + seg_gz**2) == 0)
    del seg_gx, seg_gy, seg_gz

    opnflxpos=(open_labels*0.).astype('float16') # measure of positivity
    for i in np.arange(open_labels.size):
        ri = open_labels[i]
        n_pos = np.sum(data['brr'][...,-1][np.nonzero(q_segment[...,-1]==ri)]>0) # positive area
        n_neg = np.sum(data['brr'][...,-1][np.nonzero(q_segment[...,-1]==ri)]<0) # negative area
        opnflxpos[i] = (n_pos - n_neg)/(n_pos + n_neg)

    data['q_segment']     = q_segment
    data['qsl_msk']       = qsl_msk
    data['seg_msk']       = seg_msk
    data['open_labels']   = open_labels
    data['clsd_labels']   = clsd_labels
    data['labels']        = labels
    data['opnflxpos']     = opnflxpos
    data['qsl_width'] = qsl_width
    data['reg_width'] = reg_width
    
    return data

def determine_adjacency(data=None, nr=None, nz_out=None, ny_out=None, nx_out=None, xmin=None, xmax=None, ymin=None, ymax=None, zmin=None, zmax=None, z_samp=None, solrad=None, q_dir=None, b_dir=None, glbl_mdl=None, ss_eof=None, nrm_thrsh=None, pad_ratio=None, bot_rad=None, adj_thresh=None):

    if adj_thresh==None: adj_thresh=0.5

    if (data==None or ('q_segment' not in data.keys() ) ):
        data=segment_volume(data=data, nr=nr, nz_out=nz_out, ny_out=ny_out, nx_out=nx_out, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, zmin=zmin, zmax=zmax, z_samp=z_samp, solrad=solrad, q_dir=q_dir, b_dir=b_dir, glbl_mdl=glbl_mdl, ss_eof=ss_eof, nrm_thrsh=nrm_thrsh, pad_ratio=pad_ratio, bot_rad=bot_rad)

    # let's get a mask set up with all of the regions along an axis.
    # It will be a boolean of whether any given pixel is
    # nearer to that region than the threshold

    q_segment=data['q_segment']
    nx, ny, nz = q_segment.shape
    regs=data['labels']
    nregs = regs.size
    adj_msk = np.zeros((nx, ny, nz, nregs)).astype('bool')

    # here we calculate distance transforms
    qsl_width = data['qsl_width']
    for i in np.arange(nregs):
        print('Finding distance to region '+str(regs[i]))
        dist_i = ndi.distance_transform_edt( q_segment!=regs[i] ).astype('float32')
        print('Determining proximity mask') # and compare to the threshold to create a mask.
        adj_msk[...,i] = dist_i * data['crr'].mean() <= qsl_width * data['crr'] * adj_thresh
        #reg_adj_ind[str(regs[i])] = np.nonzero(adj_msk[...,i])

    data['adj_msk']=adj_msk

    return data
        
    

def get_reg_qsl(data=None, reg_labels=None, logic=None):
    if data==None:
        print('must supply seg data')

    if reg_labels==None:
        print('must supply label (or list of labels)')
    elif np.size(reg_labels)==1: reg_labels=[reg_labels]
    else: reg_labels=list(reg_labels)
    
    if logic==None:
        if np.size(reg_labels)>1:
            print('multiple regions and qsl_type not specified ... defaulting to Union')
        logic='Union'

    #note that the adjacency mask runs over non-zero entries to the label list.

    all_labels=data['labels']
    msk = (np.ones(data['qsl_msk'].shape) * (logic=='Intrs')).astype('bool')
    for i in range(all_labels.size):
        if all_labels[i] in reg_labels:
            if logic=='Union':
                msk = msk | data['adj_msk'][...,i] # unions
            elif logic=='Intrs':
                msk = msk & data['adj_msk'][...,i] # intersections
            else: print('qsl_type must be one of "Intrs" or "Union"')

    msk = msk & data['qsl_msk'] # project against qsl mask
    
    return msk

def get_groups(data=None):

    nonempty_pairs = []
    # first we append pairwise groupings
    for l1 in data['open_labels']:
        for l2 in data['open_labels'][np.nonzero(data['open_labels']>l1)]:
            hqv = get_reg_qsl(data=data, reg_labels=(l1, l2), logic='Intrs')
            top = (np.sum(hqv[...,-1]) > 0)
            vol = np.sum(hqv)
            if vol > 0:
                print('overlap found for pair (',l1,',',l2,', vol: ',vol,', top: ',top,')')
                nonempty_pairs.append(((l1, l2), vol, top))
    # now we explore depth -- triple groups
    nonempty_trips = []
    for i in range(len(nonempty_pairs)):
        l1, l2 = nonempty_pairs[i][0]
        for j in data['open_labels'][np.nonzero(data['open_labels'] > l2)]:
            hqv = get_reg_qsl(data=data, reg_labels=(l1, l2, j), logic='Intrs')
            top = (np.sum(hqv[...,-1]) > 0)
            vol = np.sum(hqv)
            if vol > 0:
                print('overlap found for pair (',l1,',',l2,',',j,', vol: ',vol,', top: ',top,')')
                nonempty_trips.append(((l1, l2, j), vol, top))
    # now quadruple groups
    nonempty_quads = []
    for i in range(len(nonempty_trips)):
        l1, l2, l3 = nonempty_trips[i][0]
        for j in data['open_labels'][np.nonzero(data['open_labels'] > l3)]:
            hqv = get_reg_qsl(data=data, reg_labels=(l1, l2, l3, j), logic='Intrs')
            top = (np.sum(hqv[...,-1]) > 0)
            vol = np.sum(hqv)
            if vol > 0:
                print('overlap found for pair (',l1,',',l2,',',l3,',',j,', vol: ',vol,', top: ',top,')')
                nonempty_quads.append(((l1, l2, l3, j), vol, top))
    # now quadruple quints
    nonempty_quints = []
    for i in range(len(nonempty_quads)):
        l1, l2, l3, l4 = nonempty_quads[i][0]
        for j in data['open_labels'][np.nonzero(data['open_labels'] > l4)]:
            hqv = get_reg_qsl(data=data, reg_labels=(l1, l2, l3, l4, j), logic='Intrs')
            top = (np.sum(hqv[...,-1]) > 0)
            vol = np.sum(hqv)
            if vol > 0:
                print('overlap found for pair (',l1,',',l2,',',l3,',',l4,',',j,', vol: ',vol,', top: ',top,')')
                nonempty_quads.append(((l1, l2, l3, l4, j), vol, top))

    data['nonempty_pairs'] = nonempty_pairs
    data['nonempty_trips'] = nonempty_trips

    return data
                
                
            
        
        

def make_null_box(data=None, null_num=None, box_size=None):
    if data==None:
        print('must supply datacube')
        
    if null_num==None: null_num=0
    if box_size==None: box_size=2
    if ((box_size%2)>0):
        box_size+=1
        print('box must be of even size -- increasing by one')

    crr = data['crr'][0,0,:]
    cth = data['cth'][0,:,0]
    cph = data['cph'][:,0,0]
    nrr = crr.size
    nth = cth.size
    nph = cph.size

    null_locs=data['null_locs'].T
    
    rr_ind = np.argmax(crr > null_locs[null_num,2]*696)
    th_ind = np.argmax(cth > null_locs[null_num,1])
    ph_ind = np.argmax(cph > null_locs[null_num,0])

    nll_indices=[]

    hwidth = int(box_size/2)
    if hwidth==0: nll_indices.append((ph_ind, th_ind, rr_ind))
    
    for i in range(-hwidth,hwidth):
        if (0 <= ph_ind+i < nph):
            for j in range(-hwidth,hwidth):
                if (0 <= th_ind+j < nth):
                    for k in range(-hwidth,hwidth):
                        if (0 <= rr_ind+k < nrr):
                            nll_indices.append((ph_ind+i, th_ind+j, rr_ind+k))

    return nll_indices


def get_adj_nll(data=None, labels=None, logic=None, msk=None):
    if data==None:
        print('must supply seg data')

    # get qsl
    if (labels==None):
        if msk==None:
            print('must supply a mask or list of regions')
        else: qsl_msk=msk
    else:
        qsl_msk = get_reg_qsl(data=data, labels=labels, logic=logic)

    # get nulls
    if 'null_locs' in data.keys():
        null_locs=data['null_locs'].T
        null_locs=null_locs[np.argsort(null_locs[:,2]),:]
    else:
        null_locs=(np.array([0,0,0]))
        print('no nulls supplied')
    N_null=null_locs.shape[0]

    # find overlap
    adj_nll_list=[]
    for n in range(N_null):
        nll_box = make_null_box(data=data, null_num=n, box_size=2)
                    
        if np.sum(qsl_msk[coords] for coords in nll_box) > 0:
            adj_nll_list.append(labels, logic, n)

    return adj_nll_list




def get_null_reg_dist(data=None, reg_labels=None, null_list=None):
    if data==None:
        print('must supply seg data')

    # get qsl
    if (reg_labels==None):
            print('region list not supplied -- applying all')
            reg_labels = data['labels']
        
    # get nulls
    if 'null_locs' in data.keys():
        null_locs=data['null_locs'].T
        null_locs=null_locs[np.argsort(null_locs[:,2]),:]
    else:
        null_locs=(np.array([0,0,0]))
        print('no nulls supplied')
        
    N_null=null_locs.shape[0]

    # need coordinate differentials to normalize distances

    metph, metth, metrr = get_differentials(data=data)
    
    if null_list==None:
        print('null list not supplied -- applying to all')
        null_list=range(N_null)

    nll_dist_list=[]
    nll_dist_list.append(reg_labels)
    nll_dist_list.append(null_list)
    nll_dist_array = np.zeros((np.size(reg_labels), np.size(null_list)))
    nll_bool_array = np.zeros((np.size(reg_labels), np.size(null_list))).astype('bool')
    print('Getting distances')
    for i in range(np.size(reg_labels)):
        msk = get_reg_qsl(data=data, reg_labels=reg_labels[i])
        # ss_nonzero = np.nonzero(msk)
        # for n in null_list:
        #     nll_coord = make_null_box(data=data, null_num=n, box_size=2)
        #     nll_dist = np.sqrt((irr - nll_coord[0][2])**2 + (ith - nll_coord[0][1])**2 + (iph - nll_coord[0][0])**2)[ss_nonzero].min()
        #     nll_dist_array[i,n] = nll_dist
        #     print('region: '+str(reg_labels[i])+', null: '+str(n)+', distance: '+str(nll_dist))
        for j in range(np.size(null_list)):
            n = null_list[j]
            box = make_null_box(data=data, null_num=n, box_size=2)
            for corner in box:
                if data['q_segment'][corner]==reg_labels[i]: nll_bool_array[i,j]=True # does the null have a corner in the actual domain?
            # we'll do measurements in sphericals but assuming orthonormality -- this is innacurate for large distances but we only care about small distances.
            dist_rr = (data['crr'] - data['solrad']*data['null_locs'].T[n,2]) / metrr
            dist_th = data['crr']*(np.pi / 180)*(data['cth'] - data['null_locs'].T[n,1]) / metth 
            dist_ph = data['crr']*np.cos(data['cth']*np.pi/180)*(np.pi/180) * (data['cph'] - data['null_locs'].T[n,0]) / metph
            nll_dist = np.sqrt(dist_rr**2 + dist_th**2 + dist_ph**2)[np.nonzero(msk)].min()
            nll_dist_array[i,j] = nll_dist
            print('region: '+str(reg_labels[i])+', null: '+str(n)+', distance: '+str(nll_dist))

    nll_dist_list.append((nll_bool_array, nll_dist_array))
    
    return nll_dist_list
        
            



def get_nll_regs(data=None, null_list=None, msk=None):
    if data==None:
        print('must supply seg data')

    if null_list==None:
        print('must supply null indices')

    if 'null_locs' not in data.keys():
        print('data must contain null locations')

    if 'adj_msk' not in data.keys():
        adj_msk = squash_seg.determin_adjacency(data=data)
    else:
        adj_msk = data['adj_msk']

    labels=data['labels']
    null_regs = []
    for i in range(labels.size):
        for j in null_list:
            jbox=make_null_box(data=data, null_num=j, box_size=2)
            jmsk = adj_msk[...,i]
            if np.sum([jmsk[coords] for coords in jbox]) > 0:
                null_regs.append((labels[i], j))
            
    return null_regs




    

def make_cmap(data=None, shuffle_colors=None):
    if data==None:
        print('must supply seg data')

    # define a custom color for segmentation visualization
    labels = np.unique(data['q_segment'])
    openlin = np.linspace(0, 1, np.size(labels[np.argwhere(labels < 0)]) + 1)[1:]
    clsdlin = np.linspace(0, 1, np.size(labels[np.argwhere(labels > 0)]) + 1)[1:]
    if shuffle_colors==True:
        from random import shuffle
        shuffle(openlin)
        shuffle(clsdlin)
    opencolors = plt.cm.winter(openlin)
    clsdcolors = plt.cm.autumn(clsdlin)
    whitecolor = plt.cm.binary(0)
    blackcolor = plt.cm.binary(255)
    mycolors = np.vstack((np.flip(opencolors, axis=0),blackcolor,clsdcolors))
    segcmap = mpl.colors.LinearSegmentedColormap.from_list('my_colormap', mycolors)

    return segcmap

def load_data(fname=None):
    fdata = np.load(fname)
    data = {}
    for key in fdata.keys(): data[key]=fdata[key]

    return data

def save_data(fname=None, data=None):
    if (fname==None):
        print('Writing to default filename: seg_data.npz')
        fname='seg_data.npz'
    if (data==None):
        print('Error: must supply dictionary to be saved')
    else: np.savez(fname, **data)
    return
    
def export_vtk(data=None, odir=None, ofname=None, rr_rng=None, th_rng=None, ph_rng=None):

    from pyevtk.hl import gridToVTK
    import os
    
    if data==None:
        print('must supply seg data')

    if odir==None: odir='./vtk_files/'
    if os.path.exists(odir)==False: os.mkdir(odir)
    if ofname==None: ofname='slog10q_3dvis'
    
    # We'll generate vtk files for the main objects with an option to do it for a custom object.
    crr=0.      + data['crr']
    cth=np.pi/2 - data['cth']*np.pi/180.
    cph=0.      + data['cph']*np.pi/180.
    brr=data['brr']
    bth=data['bth']
    bph=data['bph']



    
    # First we need all of the coordinates in cartesian.
    cx=crr*np.cos(cph)*np.sin(cth)
    cy=crr*np.sin(cph)*np.sin(cth)
    cz=crr*np.cos(cth)

    # and the index ranges
    nrr = crr.shape[2]
    nth = crr.shape[1]
    nph = crr.shape[0]


    # Now we can export values.

    # first we determine the index range

    if rr_rng is None:
        rl, rr = 0, nrr
    else:
        rr_rng=list(i*data['solrad'] for i in rr_rng) # takes arg in units of solar radius
        rr_rng.sort()
        temp_msk = (((crr[0,0,:] >= rr_rng[0]) & (crr[0,0,:] <= rr_rng[1])))
        if temp_msk.sum()==0: print('radial clip has no volume')
        rl = np.argmax(temp_msk)
        rr = nrr - 1 - np.argmax(temp_msk[::-1])

    if th_rng is None:
        tl, tr = 0, nth
    else:
        th_rng=list(np.pi/2 - i*np.pi/180. for i in th_rng) # takes arg in degrees and converts to polar angle
        th_rng.sort()
        temp_msk = (((cth[0,:,0] >= th_rng[0]) & (cth[0,:,0] <= th_rng[1])))
        if temp_msk.sum()==0: print('latitudinal clip has no volume')
        tl = np.argmax(temp_msk)
        tr = nth - 1 - np.argmax(temp_msk[::-1])

    if ph_rng is None:
        pl, pr = 0, nph
    else:
        ph_rng=list(i*np.pi/180. for i in ph_rng) # takes arg in degrees and converts to azimuthal angle.
        ph_rng.sort()
        temp_msk = (((cph[:,0,0] >= ph_rng[0]) & (cph[:,0,0] <= ph_rng[1])))
        if temp_msk.sum()==0: print('longitudinal clip has no volume')
        pl = np.argmax(temp_msk)
        pr = nph - 1 - np.argmax(temp_msk[::-1])

    slog10q=data['slog10q']
    qsl_msk=data['qsl_msk']
    q_segment=data['q_segment']
    bx=brr*np.sin(cth)*np.cos(cph) + bth*np.cos(cth)*np.cos(cph) + bph *(-np.sin(cph))
    by=brr*np.sin(cth)*np.sin(cph) + bth*np.cos(cth)*np.sin(cph) + bph *( np.cos(cph))
    bz=brr*np.cos(cth)             + bth*(-np.sin(cth))

    gridToVTK(odir+ofname, cx[pl:pr,tl:tr,rl:rr].copy(), cy[pl:pr,tl:tr,rl:rr].copy(), cz[pl:pr,tl:tr,rl:rr].copy(),
              pointData = { "slog10q" : slog10q[pl:pr,tl:tr,rl:rr].copy(),
                            "qsl_msk" : qsl_msk[pl:pr,tl:tr,rl:rr].astype('int16').copy(),
                            "q_segment" : q_segment[pl:pr,tl:tr,rl:rr].copy(),
                            "bfield" : (bx[pl:pr,tl:tr,rl:rr].copy(), by[pl:pr,tl:tr,rl:rr].copy(), bz[pl:pr,tl:tr,rl:rr].copy())})    


    return


def visualize(data=None, rr=None, th=None, ph=None, data_key=None, mask_key=None, inv_mask=None, window=None, figsize=None):

    # get data cube
    if data_key==None: data_key='slog10q'
    if type(data)==dict:
        if type(data_key)==str:
            if data_key in data.keys():
                dcube=data[data_key]
                dmask=np.ones(dcube.shape)
            else:
                print('key not in dict')
                pdb.set_trace()
        if type(mask_key)==str:
            if mask_key in data.keys():
                dmask=data[mask_key]
                if (inv_mask==True): dmask=~dmask
        elif mask_key!=None:
            print('key must be a string')
            pdb.set_trace()
    else:
        print('data must be a dictionary')
        pdb.set_trace()
    if inv_mask==None: inv_mask=False

    if 'null_locs' in data.keys():
        P_null=data['null_locs'].T
        P_null=P_null[np.argsort(P_null[:,2]),:]
    else: P_null=(np.array([0,0,0]))
    N_null=P_null.shape[0]

    # get coordinates
    crr = data['crr'][0,0,:]/696.
    cth = data['cth'][0,:,0]
    cph = data['cph'][:,0,0]
    nrr = crr.size
    nth = cth.size
    nph = cph.size
    th_min= cth.min()
    th_max= cth.max()
    ph_min= cph.min()
    ph_max= cph.max()
    rr_min= crr.min()
    rr_max= crr.max()
    
    if window==None: window=0

    if figsize==None: figsize=(16,9)
    elif type(figsize)!=tuple:
        print('figsize must be a tuple')
        pdb.set_trace()

    if 'fig' not in locals():
        fig = plt.figure(window, figsize=figsize)
    fig.clf()

    if rr==None: rr=crr[-1]
    if th==None: th=cth[int(nth/2)]
    if ph==None: ph=cph[int(nph/2)]
    lr=np.log(rr)*1.5/np.log(2.5)+1.
        
    # get coordinate indices
    if rr <= crr.min(): irr=0
    elif rr >= crr.max(): irr=-1
    else: irr = np.argmin(crr <= rr)-1

    if th <= cth.min(): ith=0
    elif th >= cth.max(): ith=-1
    else: ith = np.argmin(cth <= th)-1
    
    if ph < cph.min(): iph=0
    elif ph > cph.max(): iph=-1
    else: iph = np.argmin(cph <= ph)-1
    
    rr = crr[irr]
    th = cth[ith]
    ph = cph[iph]
    rr0, th0, ph0 = rr, th, ph

    if data_key=='slog10q':
        ctable ='viridis'
        vrange = (-3,3)
    if data_key=='q_segment':
        ctable = make_cmap(dada=data)
        vrange = (data[data_key].min(), data[data_key].max())
    if data_key=='brr':
        ctable = 'bwr_r'
        vrange = (-1,1)

    fig, axes = plt.subplots(num=window, nrows=2, ncols=2, gridspec_kw={'height_ratios': [(th_max-th_min),1.5*180/np.pi], 'width_ratios': [1.5*180/np.pi,(ph_max-ph_min)]})
    im1 = axes[0,0].imshow((dcube*dmask)[iph,:,:]  , vmin=vrange[0], vmax=vrange[1], cmap=ctable, extent=[rr_min,rr_max,th_min,th_max], origin='lower', aspect=(np.pi/180))
    im2 = axes[0,1].imshow((dcube*dmask)[:,:,irr].T, vmin=vrange[0], vmax=vrange[1], cmap=ctable, extent=[ph_min,ph_max,th_min,th_max], origin='lower', aspect=(1.))
    im3 = axes[1,1].imshow((dcube*dmask)[:,ith,:].T, vmin=vrange[0], vmax=vrange[1], cmap=ctable, extent=[ph_min,ph_max,rr_min,rr_max], origin='lower', aspect=(180./np.pi))
    ch1a, = axes[0,0].plot([rr_min,rr_max], [th,th], '--', linewidth=1, color='black')
    ch1b, = axes[0,0].plot([lr,lr], [th_min,th_max], '--', linewidth=1, color='black')
    ch2a, = axes[0,1].plot([ph_min,ph_max], [th,th], '--', linewidth=1, color='black')
    ch2b, = axes[0,1].plot([ph,ph], [th_min,th_max], '--', linewidth=1, color='black')
    ch3a, = axes[1,1].plot([ph_min,ph_max], [lr,lr], '--', linewidth=1, color='black')
    ch3b, = axes[1,1].plot([ph,ph], [rr_min,rr_max], '--', linewidth=1, color='black')
    axes[0,0].set_ylabel(r'Latitude -- $\theta$ [deg]')
    axes[0,0].set_xlabel(r'Radius -- $log_{10}~r$')
    axes[0,0].yaxis.set_ticks(np.linspace(-60, 60, 5 ))
    axes[0,0].xaxis.set_ticks(np.linspace(1,  2.5, 2 ))
    axes[0,1].yaxis.set_ticks(np.linspace(-60, 60, 5 ))
    axes[0,1].xaxis.set_ticks(np.linspace(0,  360, 7 ))
    axes[1,1].set_xlabel(r'Longitude -- $\phi$ [deg]')
    axes[1,1].xaxis.set_ticks(np.linspace(0,  360, 7 ))
    axes[1,1].yaxis.set_ticks(np.linspace(1,  2.5, 2 ))
    #axes[1,0].text(0.5, 0.6, r'Radius -- $R$ [solar radii]', transform=axes[1,0].transAxes, horizontalalignment='center', verticalalignment='center')
    #axes[1,0].text(0.5, 0.45, r'(log scaling)', transform=axes[1,0].transAxes, horizontalalignment='center', verticalalignment='center')
    #axes[1,0].text(0.5, 0.6, r'$log_{10}$ Radius', transform=axes[1,0].transAxes, horizontalalignment='center', verticalalignment='bottom')
    axes[1,0].axis('off')
    #axes[0,0].set_title(r'$\phi =$ '+'{:2.2f}'.format(ph))
    axes[0,1].set_title(r'')
    #axes[1,1].set_title(r'$\theta =$ '+'{:2.2f}'.format(th))
    fig.subplots_adjust(right=0.8)
    ar1 = axes[0,1].annotate('', xy=(ph,  88), xytext=(-(45 + (2.5-lr)*180./np.pi),  88), arrowprops=dict(arrowstyle="<-", connectionstyle="arc,angleA=45, angleB=90, armA=21, armB=15, rad=8"), xycoords='data', textcoords='data')
    ar2 = axes[0,1].annotate('', xy=(360, th), xytext=(360,-(130 + (2.5-lr)*180./np.pi)), arrowprops=dict(arrowstyle="<-", connectionstyle="arc,angleA=45, angleB=0,  armA=21, armB=15, rad=8"), xycoords='data', textcoords='data')
    cbar_ax = fig.add_axes([0.85, 0.125, 0.05, 0.725])

    if data_key=='q_segment':
        cbar = fig.colorbar(im2, cax=cbar_ax, orientation = 'vertical', ticks=[dcube.min(), 0, dcube.max()])
        cbar.ax.set_yticklabels([r'Open', r'OCB', r'Closed'])
    else:
        cbar = fig.colorbar(im2, cax=cbar_ax, orientation = 'vertical', ticks=np.linspace(vrange[0], vrange[1], 5))
    
    axcolor = 'lightgoldenrodyellow'
    axrr = plt.axes([0.075, 0.28, 0.15, 0.03], facecolor=axcolor)
    axth = plt.axes([0.075, 0.23, 0.15, 0.03], facecolor=axcolor)
    axph = plt.axes([0.075, 0.18, 0.15, 0.03], facecolor=axcolor)
    axbp = plt.axes([0.075, 0.13, 0.05, 0.03], facecolor=axcolor)
    axbn = plt.axes([0.175, 0.13, 0.05, 0.03], facecolor=axcolor)
    axbr = plt.axes([0.125, 0.13, 0.05, 0.03], facecolor=axcolor)
    axbt = plt.axes([0.230, 0.13, 0.02, 0.03], frameon=False)
    axbt.set_xticks([])
    axbt.set_yticks([])
    rax1 = plt.axes([0.3, 0.9, 0.1, 0.08], facecolor=axcolor)
    rax2 = plt.axes([0.5, 0.9, 0.1, 0.08], facecolor=axcolor)
    rax3 = plt.axes([0.7, 0.9, 0.1, 0.08], facecolor=axcolor)
    null_label = axbt.text(0,0,'')

    draw_params={'rr':rr, 'th':th, 'ph':ph, 'mask_key':mask_key, 'data_key':data_key, 'inv_mask':inv_mask}

    def redraw():

        nonlocal draw_params
        rr = draw_params['rr']
        th = draw_params['th']
        ph = draw_params['ph']
        data_key = draw_params['data_key']
        mask_key = draw_params['mask_key']
        inv_mask = draw_params['inv_mask']
        # get coordinate indices
        if rr <= crr.min(): irr=0
        elif rr >= crr.max(): irr=-1
        else: irr = np.argmin(crr <= rr)

        if th <= cth.min(): ith=0
        elif th >= cth.max(): ith=-1
        else: ith = np.argmin(cth <= th)

        if ph < cph.min(): iph=0
        elif ph > cph.max(): iph=-1
        else: iph = np.argmin(cph <= ph)

        #rr = crr[irr]
        #th = cth[ith]
        #ph = cph[iph]
        lr=np.log(rr)*1.5/np.log(2.5)+1.
        draw_params={'rr':rr, 'th':th, 'ph':ph, 'mask_key':mask_key, 'data_key':data_key, 'inv_mask':inv_mask}

        dcube=data[data_key]
        if data_key=='slog10q':
            ctable='viridis'
            vrange = (-3,3)
        if data_key=='q_segment':
            ctable = make_cmap(data=data)
            vrange = (data[data_key].min(), data[data_key].max())
        if data_key=='brr':
            ctable = 'bwr_r'
            vrange = (-1,1)
            
        if mask_key==None: dmask=np.ones(dcube.shape)
        elif mask_key=='qsl_msk':
            if inv_mask==True: dmask = data['qsl_msk']
            else: dmask = ~data['qsl_msk']
        elif mask_key=='seg_msk':
            if inv_mask==True: dmask = ~data['seg_msk']
            else: dmask = data['seg_msk'] 

        nonlocal im1, im2, im3, ch1a, ch1b, ch2a, ch2b, ch3a, ch3b, ar1, ar2, cbar, null_label

        im1.set_cmap(ctable)
        im2.set_cmap(ctable)
        im3.set_cmap(ctable)
        
        im1.set_data((dcube*dmask)[iph,:,:])
        im2.set_data((dcube*dmask)[:,:,irr].T)
        im3.set_data((dcube*dmask)[:,ith,:].T)
        im1.set_clim(vmin=vrange[0], vmax=vrange[1])
        im2.set_clim(vmin=vrange[0], vmax=vrange[1])
        im3.set_clim(vmin=vrange[0], vmax=vrange[1])

        ch1a.set_ydata([th,th])
        ch1b.set_xdata([lr,lr])
        ch2a.set_ydata([th,th])
        ch2b.set_xdata([ph,ph])
        ch3a.set_ydata([lr,lr])
        ch3b.set_xdata([ph,ph])

        ar1.remove()
        ar2.remove()
        ar1 = axes[0,1].annotate('', xy=(ph,  88), xytext=(-(45 + (2.5-lr)*180./np.pi),  88), arrowprops=dict(arrowstyle="<-", connectionstyle="arc,angleA=45, angleB=90, armA=21, armB=15, rad=8"), xycoords='data', textcoords='data')
        ar2 = axes[0,1].annotate('', xy=(360, th), xytext=(360,-(130 + (2.5-lr)*180./np.pi)), arrowprops=dict(arrowstyle="<-", connectionstyle="arc,angleA=45, angleB=0,  armA=21, armB=15, rad=8"), xycoords='data', textcoords='data')
        null_label.remove()
        if null_vis==True:
            null_label=axbt.text(0,0.5,r'$N_i=$'+str(c_null))
        else:
            null_label=axbt.text(0,0.5,'')
        

        
        if data_key == 'q_segment':
            tickvals = [dcube.min(), 0, dcube.max()]
            ticknames = ['open', 'OCB', 'closed']
        else:
            tickvals = np.linspace(vrange[0], vrange[1], 5)
            ticknames = ['{:1.1f}'.format(i) for i in tickvals]
        
        cbar.set_ticks(tickvals)
        cbar.ax.set_yticklabels(ticknames)
        
        return 0
    
    def update_rr_coord(val):
        nonlocal draw_params, null_vis, null_ini, rr_nodraw
        draw_params['rr'] = srr.val
        if rr_nodraw==False: # behavior when resetting a single coordinate allows immediate redraw and logic
            if (null_vis==True) & (null_ini==False): null_vis=False
            null_ini=False
            redraw()
            fig.canvas.draw_idle()
        else:
            rr_nodraw=False # alternative behavior to suppress redraw and logic for multiple coordinate updates
            #print('r redraw suppressed')
            #print(null_vis)
        return

    def update_th_coord(val):
        nonlocal draw_params, null_vis, null_ini, th_nodraw
        draw_params['th'] = sth.val
        if th_nodraw==False:
            if (null_vis==True) & (null_ini==False): null_vis=False
            null_ini=False
            redraw()
            fig.canvas.draw_idle()
        else:
            th_nodraw=False
            #print('th redraw suppressed')
            #print(null_vis)
        return

    def update_ph_coord(val):
        nonlocal draw_params, null_vis, null_ini, ph_nodraw
        draw_params['ph'] = sph.val
        if ph_nodraw==False:
            if (null_vis==True) & (null_ini==False): null_vis=False
            null_ini=False
            redraw()
            fig.canvas.draw_idle()
            #print('redrawing on ph update')
            #print(null_vis)
        else:
            ph_nodraw=False
        return

    def update_mask_key(label):
        nonlocal draw_params
        if label =='HQV mask':  draw_params['mask_key']='qsl_msk'
        if label =='seg bound': draw_params['mask_key']='seg_msk'
        if label =='No mask':   draw_params['mask_key']=None
        #print(mask_key)
        redraw()
        fig.canvas.draw_idle()

    def update_data_key(label):
        nonlocal draw_params
        if label=='slog10 Q':     draw_params['data_key']='slog10q'
        if label=='seg map':      draw_params['data_key']='q_segment'
        if label=='radial field': draw_params['data_key']='brr'
        #print(data_key)
        redraw()
        fig.canvas.draw_idle()

    def update_inv_mask(label):
        nonlocal draw_params
        if label=='Inverse Mask': draw_params['inv_mask']=True
        if label=='Default Mask': draw_params['inv_mask']=False
        #print(inv_msk)
        redraw()
        fig.canvas.draw_idle()
        
    def increase_null_pos(value):
        nonlocal c_null, null_vis, null_ini, rr_nodraw, th_nodraw
        null_ini=True
        #print(null_vis)
        if (null_vis==False): null_vis=True
        else: c_null = (c_null - 1)%N_null
        #print(null_vis)
        #print(c_null)
        rr_nodraw=True
        th_nodraw=True
        srr.set_val(P_null[c_null,2])
        sth.set_val(P_null[c_null,1])
        sph.set_val(P_null[c_null,0])
        
    def decrease_null_pos(value):
        nonlocal c_null, null_vis, null_ini, rr_nodraw, th_nodraw
        null_ini=True
        #print(null_vis)
        if (null_vis==False): null_vis=True
        else: c_null = (c_null + 1)%N_null
        #print(null_vis)
        #print(c_null)
        rr_nodraw=True
        th_nodraw=True
        srr.set_val(P_null[c_null,2])
        sth.set_val(P_null[c_null,1])
        sph.set_val(P_null[c_null,0])

    def reset_coords(value):
        srr.set_val(rr0)
        sth.set_val(th0)
        sph.set_val(ph0)
        nonlocal c_null, null_vis
        null_vis=False
        c_null = N_null-1

    srr = Slider(axrr, r'$r$',       1.0, 2.5,  valinit=rr, valfmt="%2.3f")
    sth = Slider(axth, r'$\theta$', -88., 88.,  valinit=th, valfmt="%2.1f")
    sph = Slider(axph, r'$\phi$'  ,  0.0, 360., valinit=ph, valfmt="%2.1f")
    
    data_selector_button = RadioButtons(rax1, ('slog10 Q', 'seg map', 'radial field'), active=0)    
    mask_selector_button = RadioButtons(rax2, ('No mask', 'HQV mask', 'seg bound'), active=0)
    inv_maskersion_button = RadioButtons(rax3, ('Default Mask', 'Inverse Mask'), active=0)

    null_inc_button = Button(axbn, 'next null', color='w', hovercolor='b')
    null_dec_button = Button(axbp, 'prev null', color='w', hovercolor='b')
    reset_button = Button(axbr, 'reset', color='r', hovercolor='b')
    null_vis=False
    null_ini=False
    rr_nodraw=False
    th_nodraw=False
    ph_nodraw=False
    coord_cnt=0
    c_null=N_null-1

    null_inc_button.on_clicked(increase_null_pos)
    null_dec_button.on_clicked(decrease_null_pos)
    reset_button.on_clicked(reset_coords)
    
    srr.on_changed(update_rr_coord)
    sth.on_changed(update_th_coord)
    sph.on_changed(update_ph_coord)
    
    data_selector_button.on_clicked(update_data_key)
    mask_selector_button.on_clicked(update_mask_key)
    inv_maskersion_button.on_clicked(update_inv_mask)

    #rrbox.on_submit(subrr)
    #thbox.on_submit(subth)
    #phbox.on_submit(subph)

    

    return data_selector_button, mask_selector_button, inv_maskersion_button, null_inc_button, null_dec_button, reset_button
