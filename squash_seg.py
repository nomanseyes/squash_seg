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

    if nrm_thrsh==None: nrm_thrsh=3.4
    slog10q = data['slog10q']
        
    ### get the sobel derivs.
    ### normalize derivs. want r*grad so deriv scales to radius
    ### in spherical, grad is d/dr, 1/r d/dth, 1/rsinth d/dph
    ### so we want, r d/dr, d/dth, 1/sinth d/dph
    ### z deriv is already d/dln(r) eqiv r*d/dr
    ### y deriv is already d/dth
    ### x deriv is d/dph. multiply by 1/sinth. -- cos(th*pi/180) in these coords (-90 to 90).
    slq_sobel_x = ndi.sobel(np.log(10.) * slog10q, axis=0, mode='reflect') * 180 / (np.pi*(data['cph'][1,0,0] - data['cph'][0,0,0])*np.cos(data['cth']*np.pi/180))
    slq_sobel_y = ndi.sobel(np.log(10.) * slog10q, axis=1, mode='reflect') * 180 / (np.pi*(data['cth'][0,1,0] - data['cth'][0,0,0]))
    slq_sobel_z = ndi.sobel(np.log(10.) * slog10q, axis=2, mode='reflect') / np.log(data['crr'][0,0,1]/data['crr'][0,0,0])
    
    ### construct sobelov Q magnitude
    log10_sbn = np.log10(np.sqrt((10.**np.absolute(slog10q.astype('int64')))**2 + slq_sobel_x**2 + slq_sobel_y**2 + slq_sobel_z**2))
    print('Sobelov norm density constructed \n%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
    
    ### create mask where norm greater than thresh
    qsl_msk = log10_sbn > nrm_thrsh
    print('Threshold / edge mask built \n%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
    ### filter using opening closing / object/hole removal
    qsl_msk = mor.closing(qsl_msk)
    print('Mask holes removed \n%%%%%%%%%%%%%%%%%%%%%%%%%%%%')

    data['qsl_msk'] = qsl_msk
    data['log10_sbn'] = log10_sbn.astype('float32')
    
    return data








def segment_volume(data=None, nr=None, nz_out=None, ny_out=None, nx_out=None, xmin=None, xmax=None, ymin=None, ymax=None, zmin=None, zmax=None, z_samp=None, solrad=None, q_dir=None, b_dir=None, glbl_mdl=None, ss_eof=None, nrm_thrsh=None, bot_rad=None):
    
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
    phi_bound_thresh=0.75
    
    q_segment = np.zeros(slog10q.shape, dtype='int32')### initiate segmentation label array.
    # it's useful to have masks for open and closed flux. We consider undecided flux to be open.
    opn_msk = (slog10q < 0)
    if ss_eof==True: opn_msk[...,-1]=True
    cls_msk = (slog10q > 0)
    nll_msk = (~opn_msk) & (~cls_msk)

    print('Calculating distance transforms \n%%%%%%%%%%%%%%%%%%%%%%%%%')
    
    # it's also useful to have the distance from the interior of the qsl regions to their boundary, and the other way.
    qsl_dist = ndi.distance_transform_edt(~qsl_msk).astype('int16') # dist to nearest qsl
    reg_dist = ndi.distance_transform_edt( qsl_msk).astype('int16') # dist to nearest reg
    qsl_halfwidth = np.mean(reg_dist[np.nonzero( qsl_msk)]) # typical qsl distance to region
    reg_halfwidth = np.mean(qsl_dist[np.nonzero(~qsl_msk)]) # typical reg distance to qsl

    print('Performing discrete flux labeling \n%%%%%%%%%%%%%%%%%%%%%%%%%')

    # now we'll label the open flux domains, above min height.
    # For open flux we'll use a distance_from_qsl mask that varies in radius... try qsl_dist * (solrad / data['crr']) > qsl_halfwidth -> qsl_dist > 2.5 * qsl_halfwidth at top
    q_segment -= skim.label((solrad * qsl_dist > qsl_halfwidth * data['crr']) & (slog10q < 0) & (data['crr'] >= bot_rad*solrad)) # all pixels not within or adjacent to a qsl
    open_labels = np.unique(q_segment[np.nonzero(q_segment < 0)])
    # now we restrict the regions to a minimim size
    for reg in open_labels:
        tmp_msk = (q_segment == reg)
        #print('region: '+str(reg)+', volume: '+str(np.sum(tmp_msk)))
        if np.sum(tmp_msk) < (2. * qsl_halfwidth)**3: # threshold size for valid regions
            q_segment = q_segment * ~tmp_msk  # zero in mask, unchanged else.
        if np.sum(tmp_msk[...,-1]) == 0: # open domains must intersect the top boundary
            q_segment = q_segment * ~tmp_msk # zero in mask, unchanged else.

    # and we'll get the closed flux labels in the same way, also above min height.
    # For closed flux we'll also vary with radius... try qsl_dist * (solrad / data['crr']) > qsl_halfwidth -> qsl_dist -> 1.0 * qsl_halfwidth at bottom.
    q_segment += skim.label((solrad * qsl_dist > qsl_halfwidth * data['crr']) & (slog10q > 0) & (data['crr'] >= bot_rad*solrad)) # all pixels not within or adjacent to a qsl
    clsd_labels = np.unique(q_segment[np.nonzero(q_segment > 0)])
    # now we restrict the regions to a minimim size
    for reg in clsd_labels:
        tmp_msk = (q_segment == reg)
        # print('region: '+str(reg)+', volume: '+str(np.sum(tmp_msk)))
        if np.sum(tmp_msk) < (2. * qsl_halfwidth)**3: # threshold size for valid regions
            q_segment = q_segment * ~tmp_msk  # zero in mask, unchanged else.
        if np.sum(tmp_msk[...,0]) == 0: # closed domains must intersection the bottom boundary.
            q_segment = q_segment * ~tmp_msk # zero in mask, unchanged else.
            
    print('Performing watershed backfill \n%%%%%%%%%%%%%%%%%%%%%%%%%')
    log10_sbn = data['log10_sbn']
    # We'll use a couple of phases here.
    # First we grow regions using the same-type mask, which just expands open-open, close-close.
    # Second, we grow regions through opposite type, but only within a qsl, where type mixing is expected.
    # Third, we grow null regions with no preference, allowing open and closed to compete.
    stime = time.time()
    q_segment += mor.watershed(           log10_sbn, q_segment * opn_msk, mask=opn_msk, watershed_line=False) * ((q_segment==0) & opn_msk)
    print('Open flux backfill completed in '+str(int(time.time()-stime))+' seconds \n%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
    stime = time.time()
    q_segment += mor.watershed(           log10_sbn, q_segment * opn_msk, mask=qsl_msk, watershed_line=False) * ((q_segment==0) & opn_msk)
    print('Open flux qsl fill completed in '+str(int(time.time()-stime))+' seconds \n%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
    stime = time.time()
    q_segment += mor.watershed(           log10_sbn, q_segment * cls_msk, mask=cls_msk, watershed_line=False) * ((q_segment==0) & cls_msk)
    print('Clsd flux backfill completed in '+str(int(time.time()-stime))+' seconds \n%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
    stime = time.time()
    q_segment += mor.watershed(           log10_sbn, q_segment * cls_msk, mask=qsl_msk, watershed_line=False) * ((q_segment==0) & cls_msk)
    print('Clsd flux qsl fill completed in '+str(int(time.time()-stime))+' seconds \n%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
    stime = time.time()
    q_segment += mor.watershed(         reg_dist**2, q_segment,           mask=nll_msk, watershed_line=False) * ((q_segment==0) & nll_msk)
    print('Null flux backfill completed in '+str(int(time.time()-stime))+' seconds \n%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
    # There may still be unnasigned regions. But these will be outliers buried deep within opposite flux types.

    #pdb.set_trace()
    
    print('Relabeling to account for periodicity \n%%%%%%%%%%%%%%%%%%%%%%%%%%%')
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
                mi = (phb_seg[0,...]==ri)
                mj = (phb_seg[1,...]==rj)
                i_sum = np.sum( mi ) 
                j_sum = np.sum( mj )
                c_sum = np.sum( mi & mj )
                min_area = np.max( (np.pi * qsl_halfwidth**2, phi_bound_thresh * np.min( (i_sum, j_sum) )) ) # smallest of individual areas, iff larger than qsl-cross-section
                if (c_sum > min_area):
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
            else:
                break # ri is empty so no reason to continue i-based loops.
                
    # now let's relabel with integer labels removing gaps
    open_labels = -np.unique(-q_segment[np.nonzero(q_segment < 0)]) # negative gets it in reverse order
    clsd_labels = np.unique(q_segment[np.nonzero(q_segment > 0)])
    # BECAUSE the labels increase monotonically, they are guaranteed to increase faster than the index labeling
    # this is why we can get away with renaming on the fly. Because the new name is always earlier in the list than the old name.
    for i in range(0, open_labels.size):
        q_segment = -(i + 1)*(q_segment == open_labels[i]) + q_segment*(q_segment != open_labels[i])
    for i in range(0, clsd_labels.size):
        q_segment = +(i + 1)*(q_segment == clsd_labels[i]) + q_segment*(q_segment != clsd_labels[i])
        
    open_labels = np.unique(q_segment[np.nonzero(q_segment < 0)])
    clsd_labels = np.unique(q_segment[np.nonzero(q_segment > 0)])
    labels = np.unique(q_segment[np.nonzero(q_segment)])
    
    print('Finished segmenting volume \n%%%%%%%%%%%%%%%%%%%%%%%%%%%')

    # and to recover our domain boundaries
    seg_msk = (ndi.generic_gradient_magnitude(q_segment, derivative=ndi.sobel)==0)

    data['q_segment'] = q_segment
    data['qsl_msk']   = qsl_msk
    data['seg_msk']   = seg_msk
    data['open_labels'] = open_labels
    data['clsd_labels'] = clsd_labels
    data['labels'] = labels
    
    return data

def determine_adjacency(data=None, nr=None, nz_out=None, ny_out=None, nx_out=None, xmin=None, xmax=None, ymin=None, ymax=None, zmin=None, zmax=None, z_samp=None, solrad=None, q_dir=None, b_dir=None, glbl_mdl=None, ss_eof=None, nrm_thrsh=None, bot_rad=None, adj_thresh=None):

    if adj_thresh==None: adj_thresh=4

    if (data==None or ('q_segment' not in data.keys() ) ):
        data=build_qsl_msk(data=data, nr=nr, nz_out=nz_out, ny_out=ny_out, nx_out=nx_out, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, zmin=zmin, zmax=zmax, z_samp=z_samp, solrad=solrad, q_dir=q_dir, b_dir=b_dir, glbl_mdl=glbl_mdl, ss_eof=ss_eof, nrm_thrsh=nrm_thrsh, bot_rad=bot_rad)

    # let's get a mask set up with all of the regions along an axis.
    # It will be a boolean of whether any given pixel is
    # nearer to that region than the threshold

    q_segment=data['q_segment']
    nx, ny, nz = q_segment.shape
    if 'labels' not in data.keys():
        data['open_labels'] = np.unique(q_segment[np.nonzero(q_segment < 0)])
        data['clsd_labels'] = np.unique(q_segment[np.nonzero(q_segment > 0)])
        data['labels'] = np.unique(q_segment[np.nonzero(q_segment)])
    regs=data['labels']
    nregs = regs.size
    adj_msk = np.zeros((nx, ny, nz, nregs)).astype('bool')
    #reg_adj_ind = {} # dictionary of indices for regions -- built of tuples that use int64 so rather expensive
    if 'qsl_halfwidth' not in data.keys():
        reg_dist = ndi.distance_transform_edt( data['qsl_msk']).astype('int16') # dist to nearest reg
        qsl_halfwidth = np.mean(reg_dist[np.nonzero( data['qsl_msk'])]) # typical qsl distance to region
    else: qsl_halfwidth=data['qsl_halfwidth']

    # here we calculate distance transforms
    for i in np.arange(nregs):
        print('Finding distance to region '+str(regs[i]))
        dist_i = ndi.distance_transform_edt( q_segment!=regs[i] ).astype('float16')
        print('Determining proximity mask') # and compare to the threshold to create a mask.
        adj_msk[...,i] = dist_i <= qsl_halfwidth * adj_thresh
        #reg_adj_ind[str(regs[i])] = np.nonzero(adj_msk[...,i])

    data['qsl_halfwidth']=qsl_halfwidth
    data['adj_msk']=adj_msk
    # data['reg_adj_ind']=reg_adj_ind

    if 'open_labels' not in data.keys():
        open_labels = np.unique(q_segment[np.nonzero(q_segment < 0)])
        data['open_labels']=open_labels
    else: open_labels = data['open_labels']

    if 'clsd_labels' not in data.keys():
        clsd_labels = np.unique(q_segment[np.nonzero(q_segment > 0)])
        data['clsd_labels']=clsd_labels
    else: clsd_labels = data['clsd_labels']
    
    if 'opnflxpos' not in data.keys():
        opnflxpos=(open_labels!=open_labels) # this will boolean to say whether each region is [open and positive]
        for i in np.arange(open_labels.size):
            ri = open_labels[i]
            n_pos = np.sum(data['brr'][...,-1][np.nonzero(q_segment[...,-1]==ri)]>0) # positive area
            n_neg = np.sum(data['brr'][...,-1][np.nonzero(q_segment[...,-1]==ri)]<0) # negative area
            opnflxpos[i] = (n_pos > n_neg) # true if pos > neg
            data['opnflxpos'] = opnflxpos

    return data

def get_reg_qsl(data=None, labels=None, qsl_type=None):
    if data==None:
        print('must supply seg data')

    if labels==None:
        print('must supply label (or list of labels)')
    else: labels=list(labels)
    
    if qsl_type==None:
        print('qsl_type not specified ... defaulting to Union')
        qsl_type='Union'

    #note that the adjacency mask runs over non-zero entries to the label list.

    all_labels=data['labels']
    msk = (np.ones(data['qsl_msk'].shape) * (qsl_type=='Intrs')).astype('bool')
    for i in range(all_labels.size):
        if all_labels[i] in labels:
            if qsl_type=='Union':
                msk = msk | data['adj_msk'][...,i] # unions
            elif qsl_type=='Intrs':
                msk = msk & data['adj_msk'][...,i] # intersections
            else: print('qsl_type must be one of "Intrs" or "Union"')

    msk = msk & data['qsl_msk'] # project against qsl mask
    
    return msk

def get_adj_nll(data=None, labels=None, qsl_type=None, msk=None):
    if data==None:
        print('must supply seg data')

    # get qsl
    if (labels==None):
        if msk==None:
            print('must supply a mask or list of regions')
        else: qsl_msk=msk
    else:
        qsl_msk = get_reg_qsl(data=data, labels=labels, qsl_type=qsl_type)

    # get nulls
    if 'null_locs' in data.keys():
        null_locs=data['null_locs'].T
        null_locs=null_locs[np.argsort(null_locs[:,2]),:]
    else:
        null_locs=(np.array([0,0,0]))
        print('no nulls supplied')
    N_null=null_locs.shape[0]

    # find overlap
    crr = data['crr'][0,0,:]
    cth = data['cth'][0,:,0]
    cph = data['cph'][:,0,0]
    adj_nll_list=[]
    for n in range(N_null):
        #get 2x2x2 cube surrounding location
        rr_ind = np.argmax(crr > null_locs[n,2]*696)
        th_ind = np.argmax(cth > null_locs[n,1])
        ph_ind = np.argmax(cph > null_locs[n,0])
        nll_indices=[]
        for i in (-1,0):
            for j in (-1,0):
                for k in (-1,0):
                    ind = (ph_ind+i, th_ind+j, rr_ind+k)
                    if -1 not in ind: nll_indices.append(ind)
                    
        if np.sum(qsl_msk[ind[0], ind[1], ind[2]]) > 0:
            adj_nll_list.append(n)

    return adj_nll_list

        
    

def make_cmap(data=None):
    if data==None:
        print('must supply seg data')

    # define a custom color for segmentation visualization
    labels = np.unique(data['q_segment'])
    openlin = np.linspace(0, 1, np.size(labels[np.argwhere(labels < 0)]) + 1)
    clsdlin = np.linspace(0, 1, np.size(labels[np.argwhere(labels > 0)]) + 1)
    opencolors = plt.cm.winter(openlin[1:])
    clsdcolors = plt.cm.autumn(clsdlin[1:])
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

def save_data(fname=None, seg_data=None):
    if (fname==None):
        print('Writing to default filename: seg_data.npz')
        fname='seg_data.npz'
    if (seg_data==None):
        print('Error: must supply dictionary to be saved')
    else: np.savez(fname, **seg_data)
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
    N_null=P_null.shape[1]

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
        ctable = make_cmap(data)
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
    rax1 = plt.axes([0.3, 0.9, 0.1, 0.08], facecolor=axcolor)
    rax2 = plt.axes([0.5, 0.9, 0.1, 0.08], facecolor=axcolor)
    rax3 = plt.axes([0.7, 0.9, 0.1, 0.08], facecolor=axcolor)

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
            ctable = make_cmap(data)
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

        nonlocal im1, im2, im3, ch1a, ch1b, ch2a, ch2b, ch3a, ch3b, ar1, ar2, cbar

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
        else: c_null = (c_null - 1)%32
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
        else: c_null = (c_null + 1)%32
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
        c_null = -1

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
    c_null=-1

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
