# init file for squash_segmentation module and associated object classes

### metadata used for specifying default values, etc, will be an object unto itself
### the model magnetic field, calculated squashing factor, and domain shape/size/etc will be individual objects.
### these will have default generators but can be specified manually as needed.


##############################################
# definitions of classes    ##################
##############################################

class Foo(object):
    pass

class Bytefile(object):

    #############################
    # a class definition for splitting file into bytearray
    # copied from Sam Cohan (stack overflow)
    #############################

    def __init__(self, f):
        self.f = f

    def __getattr__(self, item):
        return getattr(self.f, item)

    def read(self, n):
        # print("reading total_bytes=%s" % n, flush=True)
        if n >= (1 << 31):
            buffer = bytearray(n)
            idx = 0
            while idx < n:
                batch_size = min(n - idx, 1 << 31 - 1)
                # print("reading bytes [%s,%s)..." % (idx, idx + batch_size), end="", flush=True)
                buffer[idx:idx + batch_size] = self.f.read(batch_size)
                # print("done.", flush=True)
                idx += batch_size
            return buffer
        return self.f.read(n)

    def write(self, buffer):
        n = len(buffer)
        print("writing total_bytes=%s..." % n, flush=True)
        idx = 0
        while idx < n:
            batch_size = min(n - idx, 1 << 31 - 1)
            print("writing bytes [%s, %s)... " % (idx, idx + batch_size), end="", flush=True)
            self.f.write(buffer[idx:idx + batch_size])
            print("done.", flush=True)
            idx += batch_size


    

class Inputs(object):

    ###########################
    # a class object for holding initialization options and doing some default setup
    ###########################

    def __init__(self, nrr=None, nth=None, nph=None, phmin=None, phmax=None, thmin=None, thmax=None, rrmin=None, \
                 rrmax=None, r_samp=None, solrad=None, q_dir=None, b_dir=None, glbl_mdl=None, ss_eof=None, \
                 sbn_thrsh=None, ltq_thrsh=None, adj_thrsh=None, phi_bnd_thrsh=None, pad_ratio=None, bot_rad=None, \
                 auto_imp=None, auto_seg=None, protocol=None):

        self.nrr=nrr
        self.nth=nth
        self.nph=nph
        self.phmin=phmin
        self.phmax=phmax
        self.thmin=thmin
        self.thmax=thmax
        self.rrmin=rrmin
        self.rrmax=rrmax
        self.r_samp=r_samp
        self.solrad=solrad
        self.q_dir=q_dir
        self.b_dir=b_dir
        self.glbl_mdl=glbl_mdl
        self.ss_eof=ss_eof
        self.sbn_thrsh=sbn_thrsh
        self.ltq_thrsh=ltq_thrsh
        self.adj_thrsh=adj_thrsh
        self.phi_bnd_thrsh=phi_bnd_thrsh
        self.pad_ratio=pad_ratio
        self.bot_rad=bot_rad
        self.auto_imp=auto_imp
        self.auto_seg=auto_seg
        self.protocol=protocol

        ####################################################################
        # we'll keep a copy of the original inputs against future changes ##
        ####################################################################

        self.initialization = self
        del self.initialization.initialization # just to avoid recursions


    ###########################
    # end of class definition #
    ###########################


            
class Source(object):

    ##################################################################
    # a class object for tholding the source data and numerical grid #
    ##################################################################

    def __init__(self, crr=None, cth=None, cph=None, brr=None, bth=None, bph=None, slog10q=None):

        # these are the basic model data: coordinates, bfield, slog10q

        self.crr=crr
        self.cth=cth
        self.cph=cph

        self.brr=brr
        self.bth=bth
        self.bph=bph

        self.slog10q=slog10q

        # we will also keep the local coordinate gradients

        self.metrr=None
        self.metth=None
        self.metph=None

        # and the locations of any nulls in the domain

        self.null_locs=None

    ###########################
    # end of class definition #
    ###########################


class Result(object):

    ####################################################################
    # a class object for tholding the results of the segmentation etc. #
    ####################################################################

    def __init__(self, hqv_msk=None, GlnQp=None, reg_width=None, hqv_width=None, pad_msk=None, \
                 vol_seg=None, seg_msk=None, adj_msk=None, groupings=None, null_to_reg_dist=None, \
                 open_labels=None, clsd_labels=None, opos_labels=None, labels=None):

        # and here are the new attributes, which are masks, intiger arrays, and lists of strings indicating groupings.

        # these come from the mask building

        self.hqv_msk            =hqv_msk
        self.GlnQp              =GlnQp

        # these attributes come from the original segmentation

        self.reg_width          =reg_width
        self.hqv_width          =hqv_width
        self.pad_msk            =pad_msk
        self.vol_seg            =vol_seg
        self.seg_msk            =seg_msk
        self.labels             =labels
        self.open_labels        =open_labels
        self.clsd_labels        =clsd_labels
        self.opos_labels        =opos_labels

        # these come from post-processing on the various domain connectivities

        self.adj_msk            =adj_msk
        self.adj_msk_shape      =None
        self.adj_msk_boolsize   =None
        self.groupings          =groupings
        self.null_to_reg_dist   =null_to_reg_dist


    ###########################
    # end of class definition #
    ###########################









class Model(object):

    # initialize the wrapper, with the various data objects as attributes
    # attributes are the initializtion arguments, the model data, and the post-processing results.
    # methods of this class will act on these objects.
    
    def __init__(self, model=None, inputs=None, source=None, result=None, auto_import=None, auto_segment=None, auto_group=None, do_all=None, auto_save=None):

        self.inputs=Inputs()
        self.source=Source()
        self.result=Result()

        
        # here we set the input source and result input objects from attributes of the model object, unless superceeded.
        if model is not None:
            if inputs is None:
                try:
                    inputs=model.inputs
                except AttributeError:
                    print('model has no input object')
            else:
                print('Prescribed inputs used in lieu of model.inputs')
            if source is None:
                try:
                    source=model.source
                except AttributeError:
                    print('model has no source object')
            else:
                print('Prescribed source used in lieu of model.source')
            if result is None:
                try:
                    result=model.result
                except AttributeError:
                    print('model has no result object')
            else:
                print('Prescribed result used in lieu of model.result')

        # here we populate the self.inputs etc from the inputs etc, if possible.
        if inputs is not None:
            try:
                for key in inputs.__dict__.keys():
                    if key[0] != '_':
                        setattr(self.inputs, key, getattr(inputs, key))
            except AttributeError:
                print('could not populate model.inputs')

        if source is not None:
            try:
                for key in source.__dict__.keys():
                    if key[0] != '_':
                        setattr(self.source, key, getattr(source, key))
            except AttributeError:
                print('could not populate model.source')

        if result is not None:
            try:
                for key in result.__dict__.keys():
                    if key[0] != '_':
                        setattr(self.result, key, getattr(result, key))
            except AttributeError:
                print('could not populate model.result')

        if do_all:
            auto_import=True
            auto_segment=True
            auto_group=True

        if auto_import: self.do_import()
        if auto_segment: self.do_segment()
        if auto_group: self.do_group()

        if auto_save:
            self.save_data()
            
    def do_import(self):
        self.build_grid()
        self.import_squash_data()
        self.import_bfield_data()

    def do_segment(self):
        self.build_hqv_msk()
        self.segment_volume()

    def do_group(self):
        self.determine_adjacency()
        self.get_groups()
        self.get_null_reg_dist()

    def do_all(self):
        self.do_import()
        self.do_segment()
        self.do_group()

        
        

    #################################################
    # definitions of model methods ##################
    #################################################
    
    def cloan(self, donor):
        # a wrapper for porting all attributes
        for key in donor.__dict__.keys():
            if key in ['inputs', 'source', 'result']:
                portattr(self, key, getattr(donor, key))

    #################################################
    # methods for generating source #################
    #################################################


    def build_grid(self):

        ###############################################
        # a method for generating the coordinate grid #
        ###############################################

        import numpy as np
        
        if not self.inputs.nrr: self.inputs.nrr=120
        if not self.inputs.nth: self.inputs.nth = self.inputs.nrr * 4
        if not self.inputs.nph: self.inputs.nph = self.inputs.nrr * 8
        if not self.inputs.solrad: self.inputs.solrad = np.float32(696.)
        self.inputs.nrr=np.int32(self.inputs.nrr)
        self.inputs.nth=np.int32(self.inputs.nth)
        self.inputs.nph=np.int32(self.inputs.nph)
        phmin=self.inputs.phmin
        phmax=self.inputs.phmax
        thmin=self.inputs.thmin
        thmax=self.inputs.thmax
        rrmin=self.inputs.rrmin
        rrmax=self.inputs.rrmax
        r_samp=self.inputs.r_samp
        if not phmin: phmin   =  0.
        if not phmax: phmax   =  360.   
        if not thmin: thmin   = -88.   
        if not thmax: thmax   =  88.   
        if not rrmin: rrmin   =  0.0
        if not rrmax: rrmax   =  1044. # z in height above solar surface
        if not r_samp: r_samp='log'
        
        def r_sampler(r1): # must be consistent with other routines
            r=float(r1)
            if r_samp=='linear':
                r=r*(rrmax-rrmin)+rrmin
            if r_samp=='svet_exp':
                r=(np.exp(((70.+r*300.)/370.)**3)-np.exp((70./370.)**3))/(np.exp(1.)-np.exp((70./370.)**3))*(rrmax-rrmin)+rrmin
            if r_samp=='log':
                r = (self.inputs.solrad+rrmax)**r * (self.inputs.solrad+rrmin)**(1-r) - self.inputs.solrad
            return r

        Lph      = phmax-phmin
        Lth      = thmax-thmin

        r_sampler=np.vectorize(r_sampler)
        rr=r_sampler(np.array(range(self.inputs.nrr))/(self.inputs.nrr-1)) + self.inputs.solrad
        th=((np.array(range(self.inputs.nth)))/(self.inputs.nth-1.0)*Lth+thmin)
        ph=((np.array(range(self.inputs.nph)))/(self.inputs.nph-1.0)*Lph+phmin)

        print('Generating Coordinate Grid')
        self.source.crr=np.repeat(np.repeat([[rr]],self.inputs.nth,axis=1),self.inputs.nph,axis=0).astype('float32')
        self.source.cth=np.repeat(np.repeat([[th]],self.inputs.nph,axis=1),self.inputs.nrr,axis=0).transpose((1,2,0)).astype('float32')
        self.source.cph=np.repeat(np.repeat([[ph]],self.inputs.nth,axis=1),self.inputs.nrr,axis=0).transpose((2,1,0)).astype('float32')

        print('Calculating Local Metrics')
        # now we construct the metrics of the coordinates system
        self.source.metrr = (np.gradient(self.source.crr, axis=2)).astype('float32')
        self.source.metth = (np.gradient(self.source.cth, axis=1) * np.pi/180 * self.source.crr).astype('float32')
        self.source.metph = (np.gradient(self.source.cph, axis=0) * np.pi/180 * self.source.crr * np.cos(self.source.cth * np.pi/180)).astype('float32')

        print('Success  \n%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
    


        ########################
        # end of method ########
        ########################

    


    def import_squash_data(self):

        ##################################################
        # this routine imports the squashing factor data #
        ##################################################

        if not self.inputs.nrr:
            self.build_grid()
        
        import numpy as np
        import pandas as pd

        # allows generalization for other data formats
        if not self.inputs.protocol: self.inputs.protocol='standard_QSLsquasher'

        # import assuming typical data format
        if self.inputs.protocol=='standard_QSLsquasher':
            q_dir=self.inputs.q_dir
            if not q_dir: q_dir='./'
            print('Loading Q data')
            self.source.slog10q=(np.array(pd.read_table(q_dir+'/grid3d.dat',header=None).astype('float32')))[...,0].reshape((self.inputs.nph,self.inputs.nth,self.inputs.nrr))
            print('Success        \n%%%%%%%%%%%%%%%%%%%%%%%%%%%%')

        
        #########################
        # now some data cleanup #
        #########################

        # do we demand that the source surface is 'open'?

        if (self.inputs.ss_eof): self.source.slog10q[...,-1] = -np.absolute(self.source.slog10q[...,-1])


        ########################
        # end of method ########
        ########################




    def import_bfield_data(self):

        ############################################################################
        # this routine imports the magnetic field and interpolates onto the grid  ##
        ############################################################################

        if not self.inputs.nrr:
            self.build_grid()
        
        from scipy.interpolate import RegularGridInterpolator as RGI
        import numpy as np
        import pandas as pd
        import os

        # allows generalization for other data formats
        if not self.inputs.protocol: self.inputs.protocol='standard_QSLsquasher'

        # import assuming typical data format
        if self.inputs.protocol=='standard_QSLsquasher':
            b_dir=self.inputs.b_dir
            q_dir=self.inputs.q_dir
            if not b_dir:
                if not q_dir: b_dir='./'
                else: b_dir=q_dir+'/bfield_data/'
            print('Loading B data')
            # these are the native coordinates of the magnetic field grid
            ph_mag = np.array(pd.read_table(b_dir+'/xs0.dat',header=None))[...,0].astype('float32')
            th_mag = np.array(pd.read_table(b_dir+'/ys0.dat',header=None))[...,0].astype('float32')
            rr_mag = np.array(pd.read_table(b_dir+'/zs0.dat',header=None))[...,0].astype('float32') * self.inputs.solrad
            nx=ph_mag.size
            ny=th_mag.size
            nz=rr_mag.size
            # these are the components of the magnetic field on the native grid
            # minus in b_theta is due to Griffiths' def
            b_ph_nat =   (np.array(pd.read_table(b_dir+'/bx0.dat',header=None))[...,0].reshape((nz,ny,nx)) ).transpose((2,1,0)).astype('float32')
            b_th_nat = - (np.array(pd.read_table(b_dir+'/by0.dat',header=None))[...,0].reshape((nz,ny,nx)) ).transpose((2,1,0)).astype('float32')
            b_rr_nat =   (np.array(pd.read_table(b_dir+'/bz0.dat',header=None))[...,0].reshape((nz,ny,nx)) ).transpose((2,1,0)).astype('float32')
            # this is a check that the magnetic field is not corrupt
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
            if self.inputs.glbl_mdl!=False:
                if ph_mag.max() != 360:
                    nx += 1
                    ph_mag = np.append(ph_mag, 360.)
                    # 360 is now explicitly on the grid (if it wasn't before)
                    b_ph_nat = np.append(b_ph_nat, b_ph_nat[0,:,:].reshape(1,ny,nz), axis=0)
                    b_th_nat = np.append(b_th_nat, b_th_nat[0,:,:].reshape(1,ny,nz), axis=0)
                    b_rr_nat = np.append(b_rr_nat, b_rr_nat[0,:,:].reshape(1,ny,nz), axis=0)
                    ##### b_ph is now explicitly periodic and spans the entire domain.
                if (th_mag.max() != +90) and (th_mag.min() != -90):
                    ny += 2
                    th_mag = np.append(np.append(-90, th_mag), 90)
                    # the poles are now explicitly on the grid
                    ### set the value at the poles equal to the mean of the next cell row. 
                    b_ph_nat = np.append(np.append(np.mean(b_ph_nat[:,0,:], axis=0)*np.ones((nx,1,1)), b_ph_nat, axis=1), np.mean(b_ph_nat[:,-1,:], axis=0)*np.ones((nx,1,1)), axis=1)
                    b_th_nat = np.append(np.append(np.mean(b_th_nat[:,0,:], axis=0)*np.ones((nx,1,1)), b_th_nat, axis=1), np.mean(b_th_nat[:,-1,:], axis=0)*np.ones((nx,1,1)), axis=1)
                    b_rr_nat = np.append(np.append(np.mean(b_rr_nat[:,0,:], axis=0)*np.ones((nx,1,1)), b_rr_nat, axis=1), np.mean(b_rr_nat[:,-1,:], axis=0)*np.ones((nx,1,1)), axis=1)

            print('Interpolating B onto Q grid')

            b_coords = [ph_mag, th_mag, rr_mag]
            q_pts = np.stack(( self.source.cph, self.source.cth, self.source.crr )).reshape((3, self.source.cph.size)).T
    
            bph_interpolator = RGI(points=b_coords, values=b_ph_nat)
            bth_interpolator = RGI(points=b_coords, values=b_th_nat)
            brr_interpolator = RGI(points=b_coords, values=b_rr_nat)
 
            self.source.bph = bph_interpolator(q_pts).T.reshape(self.source.cph.shape).astype('float32')
            self.source.bth = bth_interpolator(q_pts).T.reshape(self.source.cph.shape).astype('float32')
            self.source.brr = brr_interpolator(q_pts).T.reshape(self.source.cph.shape).astype('float32')

            print('Importing Null Locations \n%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
            null_loc_fname = b_dir+'/nullpositions.dat'
            if os.path.isfile(null_loc_fname):
                from scipy.io import readsav
                null_dict=readsav(null_loc_fname, python_dict=True)
                self.source.null_locs=null_dict['nulls'].astype('float32')
                print('Null list imported.')
            else: print('Null list not found!')
            
            print('Success              \n%%%%%%%%%%%%%%%%%%%%%%%%%%%%')



        ########################
        # end of method ########
        ########################
    



    #################################################
    # methods for generating results ################
    #################################################



    
    def build_hqv_msk(self, sbn_thrsh=None, ltq_thrsh=None, exclude_clip=None):

        ######################################################################
        # this method generates the mask that is used for the domain mapping #
        ######################################################################
        
        import numpy as np
        from skimage import morphology as mor
        
        ### get the derivs.
        ### normalize derivs. want r*grad so deriv scales to radius
        ### in spherical, grad is d/dr, 1/r d/dth, 1/rsinth d/dph
        print('Calculating derivatives')
        ### we'll clip the datavalue before differentiating so that 10**value is within the range of float32
        clip_slog10q = np.sign(self.source.slog10q)*np.clip(np.absolute(self.source.slog10q), np.log10(2), np.log10(np.finfo('float32').max)-1)
        dslq_dph, dslq_dth, dslq_drr = np.gradient(np.log(10.) * clip_slog10q, np.pi*self.source.cph[:,0,0]/180, np.pi*self.source.cth[0,:,0]/180, self.source.crr[0,0,:], axis=(0,1,2))
        gdn_slq_ph = dslq_dph / np.cos(np.pi * self.source.cth / 180)
        del dslq_dph
        gdn_slq_th = dslq_dth
        del dslq_dth
        gdn_slq_rr = dslq_drr * self.source.crr
        del dslq_drr
        absQp = 10.**np.absolute(clip_slog10q)
    
        # ### construct grad Q magnitude
        self.result.GlnQp = np.sqrt(gdn_slq_ph**2 + gdn_slq_th**2 + gdn_slq_rr**2)
        del gdn_slq_ph, gdn_slq_th, gdn_slq_rr, clip_slog10q
        print('Gradient norm constructed')

    
        ### create mask where norm greater than thresh
        ### empirically it seems that GlnQp goes as r^(-1/2) in the volume, so we'll incorporate this into the mask.
    
        SBN_val = absQp + (self.source.crr / self.source.crr.min()) * self.result.GlnQp**2 ## sets threshold at outer radius and then scales inward.

        # threshold on just Q
        if not ltq_thrsh:
            if not self.inputs.ltq_thrsh:
                self.inputs.ltq_thrsh = 4 # can't get an RMS of Q b.c. the norm is infinite by definition... just choose something large.
                print('LTQ threshold not specified: defaulting to',self.inputs.ltq_thrsh)
            else:
                print('LTQ threshold specified by inputs obj as: ', self.inputs.ltq_thrsh)
        else:
            print('LTQ tThreshold specified directly as: ', ltq_thrsh)
            self.inputs.ltq_thrsh=ltq_thrsh
        self.inputs.ltq_thrsh=np.float32(self.inputs.ltq_thrsh)

        # threshold on SBN
        if not sbn_thrsh:
            if not self.inputs.sbn_thrsh:
                self.inputs.sbn_thrsh=1.5*np.sqrt((np.log10(SBN_val - absQp + 2)**2).mean())
                # here we've subtracted absQp (which is occasionally infinite) and replaced it with 2 (theoretical minimum)
                # so this is basically grad ln Q ^ 2 + 2, just to get a useful threshold. 
                # factor of two for geometric quadrature of Q and GQ. 
                print('SBN threshold not specified: defaulting to 1.5 x RMS ( =',self.inputs.sbn_thrsh,')')
            else:
                print('SBN threshold specified by inputs obj as: ', self.inputs.sbn_thrsh)
        else:
            print('SBN threshold specified directly as: ', sbn_thrsh)
            self.inputs.sbn_thrsh=sbn_thrsh
        self.inputs.sbn_thrsh = np.float32(self.inputs.sbn_thrsh)
            
        ltq_msk = np.log10(SBN_val) > self.inputs.ltq_thrsh
        sbn_msk = np.log10(SBN_val) > self.inputs.sbn_thrsh
        self.result.hqv_msk = ltq_msk | sbn_msk
            
        print('Threshold / edge mask built')
        ### filter using opening closing / object/hole removal
        self.result.hqv_msk = mor.closing(self.result.hqv_msk)
        print('Mask holes removed')
        self.result.GlnQp = np.clip(self.result.GlnQp, 0, 10**10).astype('float32')

        print('Success              \n%%%%%%%%%%%%%%%%%%%%%%%%%%%%')

        ########################
        # end of method ########
        ########################





    def segment_volume(self):

        #########################################################
        # this method generates the domain map in the volume ####
        # metadata for the domain map are also generated ########
        #########################################################
        import pdb
        
        if self.result.hqv_msk is None: self.build_hqv_msk()
            
        import time
        import numpy as np
        import skimage as ski
        from skimage import morphology as mor
        import skimage.measure as skim
        from scipy import ndimage as ndi

        # now we'll set up our regions for labeling and growing
        # first we do a simple discrete labeling, with a conservative mask
        # then we use region growing to backfill through the mask.

        print('Beginning volume segmentation')
    
        if not self.inputs.bot_rad: self.inputs.bot_rad=np.float32(1.0)

        nph, nth, nrr = tuple(np.float32(self.result.hqv_msk.shape))
    
        self.result.vol_seg = np.zeros(self.source.slog10q.shape, dtype='int32')### initiate segmentation label array.
        # it's useful to have masks for open and closed flux. We consider undecided flux to be open.
        opn_msk = (self.source.slog10q < 0)
        if self.inputs.ss_eof: opn_msk[...,-1]=True
        cls_msk = (self.source.slog10q > 0)
        nll_msk = (~opn_msk) & (~cls_msk)

        print('Calculating distance transforms')

        # it's also useful to have the distance from the interior of the hqv regions to their boundary, and the other way.
        hqv_dist = ndi.distance_transform_edt(~self.result.hqv_msk).astype('float32') # distance to nearest hqv
        self.result.reg_width = np.mean(4.*hqv_dist[np.nonzero(~self.result.hqv_msk)]).astype('float32') # full width is 4 times average distance to boundary
        print('Vol width: ',self.result.reg_width)
        reg_dist = ndi.distance_transform_edt( self.result.hqv_msk).astype('float32') # distance to nearest low q region
        self.result.hqv_width = np.mean(4.*reg_dist[np.nonzero( self.result.hqv_msk)]).astype('float32') # full width is 4 times average distance to boundary
        print('HQV width: ',self.result.hqv_width)
        del reg_dist

        print('Performing discrete flux labeling')

        if not self.inputs.pad_ratio: self.inputs.pad_ratio=0.25 # default is pad out to a typical quarter-width.

        # now we'll label the open flux domains, above min height.
        # we'll pad the hqv mask by a distance proportionate to the hqv_halfwidth, including a radial scaling to accomodate thicker hqvs at larger radii
        self.result.pad_msk = ( hqv_dist > self.inputs.pad_ratio * self.result.hqv_width * (self.source.crr / self.source.crr.mean()) ) # empirical radial dependence...
        self.result.vol_seg -= (skim.label(self.result.pad_msk & (self.source.slog10q < 0) & (self.source.crr >= self.inputs.bot_rad * self.inputs.solrad))).astype('int32') # all pixels not within or adjacent to a hqv
        self.result.open_labels = np.unique(self.result.vol_seg[np.nonzero(self.result.vol_seg < 0)])
        # and we'll get the closed flux labels in the same way, also above min height.
        self.result.vol_seg += skim.label(self.result.pad_msk & (self.source.slog10q > 0) & (self.source.crr >= self.inputs.bot_rad * self.inputs.solrad)) # all pixels not within or adjacent to a hqv
        self.result.clsd_labels = np.unique(self.result.vol_seg[np.nonzero(self.result.vol_seg > 0)])

        if not self.inputs.glbl_mdl: self.inputs.glbl_mdl=True
        if self.inputs.glbl_mdl:
            
            print('Associating domains across ph=0 boundary')

            if not self.inputs.phi_bnd_thrsh: self.inputs.phi_bnd_thrsh=0.5

            # now we'll associate regions across the phi boundary
            # need the labels on the boundary
            phb_seg = np.roll(self.result.vol_seg, 1, axis=0)[0:2,...] # this is just the 0,-1 columns stacked together
            phb_regs = np.unique(phb_seg[np.nonzero(phb_seg)]) # these are the regions associated with the boundary
            # and we need to know if the open regions are positive or negative to avoid accidental mixing.
            opnflxpos=(phb_regs!=phb_regs) # this will boolean to say whether each region is [open and positive]
            for i in np.arange(phb_regs.size):
                ri = phb_regs[i]
                if ri < 0: # test if open
                    n_pos = np.sum(self.source.brr[...,-1][np.nonzero(self.result.vol_seg[...,-1]==ri)]>0) # positive area
                    n_neg = np.sum(self.source.brr[...,-1][np.nonzero(self.result.vol_seg[...,-1]==ri)]<0) # negative area
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
                        mi = ((phb_seg[0,...]==ri) & (self.result.hqv_msk[0,...]==0))
                        mj = ((phb_seg[1,...]==rj) & (self.result.hqv_msk[1,...]==0))
                        i_area = np.sum( mi ) 
                        j_area = np.sum( mj )
                        c_area = np.sum( (mi & mj) )
                        if (c_area > 0):
                            # compare c_area to min allowable: composed of the larger of ( local weighted hqv cross section , smaller of ( fraction of local individual areas))
                            hqv_local_area_ave = np.sum( ( np.pi * (0.5*self.result.hqv_width)**2 * self.source.crr[0,...]**2 / self.source.crr.mean()**2 ) * (mi & mj) ) / c_area
                            if c_area > np.max(( hqv_local_area_ave, self.inputs.phi_bnd_thrsh * np.min(( i_area, j_area )) )):        
                                swap[i,j]=True

            # now the actual swapping
            swap = swap | swap.T     # first we consolidate the swap map to the minor diagonal
            for i in np.arange(nregs):
                ri = phb_regs[i]
                i_swapped=False # this region hasn't been updated yet
                for j in np.arange(i+1, nregs): # previous entries are redundant.
                    rj = phb_regs[j]
                    if (not i_swapped):
                        if swap[i,j]:
                            self.result.vol_seg[np.nonzero(self.result.vol_seg==ri)]=rj # i label -> j label, which hasn't be processed
                            swap[j,:]=swap[i,:] # j entries inherit swapability from i entries. 
                            i_swapped=True # ri has been swapped up to rj

            # some cleanup
            del phb_seg, phb_regs, opnflxpos, nregs, swap, ri, iopos, rj, jopos, mi, mj, i_area, j_area, c_area, hqv_local_area_ave, i_swapped

        print('Removing domains with sub-minimum volume')
        for reg in np.unique(self.result.vol_seg[np.nonzero(self.result.vol_seg)]):
            tmp_msk = (self.result.vol_seg == reg)
            #print('region: '+str(reg)+', volume: '+str(np.sum(tmp_msk)))
            if np.sum(tmp_msk * self.source.crr.mean()**2 / self.source.crr**2) < (0.5*self.result.hqv_width)**3: # threshold size for valid regions -- threshold increases quadratically with height to allow smaller closed domains
                self.result.vol_seg = self.result.vol_seg * ~tmp_msk  # zero in mask, unchanged else.
        del tmp_msk

        print('Performing watershed backfill into HQV padding')
        absQp = np.clip(np.absolute(self.source.slog10q), np.log10(2), 10).astype('float32')
        log10_SNQ =  np.log10(absQp + (self.source.crr / self.source.crr.max()) * self.result.GlnQp**2).astype('float32')
        del absQp

        # Initial watershed phase
        # First we grow regions into the padding layer outside the hqv mask
        stime = time.time()
        self.result.vol_seg += (mor.watershed(           log10_SNQ, self.result.vol_seg * opn_msk, mask=(opn_msk & ~self.result.hqv_msk), watershed_line=False) * ((self.result.vol_seg==0) & opn_msk & ~self.result.hqv_msk)).astype('int32')
        print('Open flux backfill completed in '+str(int(time.time()-stime))+' seconds')
        stime = time.time()
        self.result.vol_seg += (mor.watershed(           log10_SNQ, self.result.vol_seg * cls_msk, mask=(cls_msk & ~self.result.hqv_msk), watershed_line=False) * ((self.result.vol_seg==0) & cls_msk & ~self.result.hqv_msk)).astype('int32')
        print('Clsd flux backfill completed in '+str(int(time.time()-stime))+' seconds')

        print('Enforcing boundary connectivity')
        # And we require that all regions are associated with a boundary
        # These loops are split up to allow for different definitions between open and closed domains
        for reg in self.result.open_labels:
            tmp_msk = (self.result.vol_seg == reg)
            if np.sum(tmp_msk[...,-1]) == 0: # open domains must intersect the top boundary
                self.result.vol_seg = self.result.vol_seg * ~tmp_msk # zero in mask, unchanged else.
        for reg in self.result.clsd_labels:
            tmp_msk = (self.result.vol_seg == reg)
            if np.sum(tmp_msk[...,0]) == 0: # closed domains must intersection the bottom boundary.
                self.result.vol_seg = self.result.vol_seg * ~tmp_msk # zero in mask, unchanged else.
                del tmp_msk



        print('Performing restricted watershed backfill into HQV mask')
        # Second, we grow regions using the same-type mask, which just expands open-open, close-close.
        stime = time.time()
        self.result.vol_seg += (mor.watershed(           log10_SNQ, self.result.vol_seg * opn_msk, mask=opn_msk, watershed_line=False) * ((self.result.vol_seg==0) & opn_msk)).astype('int32')
        print('Open flux backfill completed in '+str(int(time.time()-stime))+' seconds')
        stime = time.time()
        self.result.vol_seg += (mor.watershed(           log10_SNQ, self.result.vol_seg * cls_msk, mask=cls_msk, watershed_line=False) * ((self.result.vol_seg==0) & cls_msk)).astype('int32')
        print('Clsd flux backfill completed in '+str(int(time.time()-stime))+' seconds')



        print('Performing transparent watershed backfill into HQV mask')
        # Third, we grow regions through opposite type, but only within a hqv, where type mixing is expected.
        stime = time.time()
        self.result.vol_seg += (mor.watershed(           log10_SNQ, self.result.vol_seg * opn_msk, mask=self.result.hqv_msk, watershed_line=False) * ((self.result.vol_seg==0) & opn_msk)).astype('int32')
        print('Open flux backfill completed in '+str(int(time.time()-stime))+' seconds')
        stime = time.time()
        self.result.vol_seg += (mor.watershed(           log10_SNQ, self.result.vol_seg * cls_msk, mask=self.result.hqv_msk, watershed_line=False) * ((self.result.vol_seg==0) & cls_msk)).astype('int32')
        print('Clsd flux backfill completed in '+str(int(time.time()-stime))+' seconds')



        print('Performing watershed backfill into residual domains')
        # Finally, we grow null regions with no preference, allowing open and closed to compete.
        stime = time.time()
        self.result.vol_seg += (mor.watershed(           1/(1 + hqv_dist), self.result.vol_seg,                         watershed_line=False) * ((self.result.vol_seg==0))).astype('int32')
        print('Final flux backfill completed in '+str(int(time.time()-stime))+' seconds')
        # There may still be unnasigned regions. But these will be outliers buried deep within opposite flux types.


        
        print('Relabeling to remove obsolete domains') 
        # now let's relabel with integer labels removing gaps
        self.result.open_labels = -np.unique(-self.result.vol_seg[np.nonzero(self.result.vol_seg < 0)]).astype('int32') # negative gets it in reverse order
        self.result.clsd_labels =  np.unique( self.result.vol_seg[np.nonzero(self.result.vol_seg > 0)]).astype('int32')
        # we want this to be a random reordering so we have to keep track of swapped regions to avoid multiple swaps.
        open_relabel = np.arange(0, self.result.open_labels.size).astype('int32')
        clsd_relabel = np.arange(0, self.result.clsd_labels.size).astype('int32')
        np.random.seed(open_relabel.size) # repeatable random seed
        np.random.shuffle(open_relabel) # random shuffle of domain order
        np.random.seed(clsd_relabel.size) # repeatable random seed
        np.random.shuffle(clsd_relabel) # random shuffle of domain order

        swapped = (self.result.vol_seg != self.result.vol_seg) # boolean to track already swapped domains

        for i in range(open_relabel.size):
            swap_msk = ((self.result.vol_seg == self.result.open_labels[i]) & ~swapped)
            self.result.vol_seg = (self.result.vol_seg * (~swap_msk) - (open_relabel[i]+1) * (swap_msk)).astype('int32')
            swapped = swapped | swap_msk

        for i in range(clsd_relabel.size):
            swap_msk = ((self.result.vol_seg == self.result.clsd_labels[i]) & ~swapped)
            self.result.vol_seg = (self.result.vol_seg * (~swap_msk) + (clsd_relabel[i]+1) * (swap_msk)).astype('int32')
            swapped = swapped | swap_msk
            
        del swapped, swap_msk, open_relabel, clsd_relabel
        self.result.open_labels = np.unique(self.result.vol_seg[np.nonzero(self.result.vol_seg  < 0)]).astype('int32')
        self.result.clsd_labels = np.unique(self.result.vol_seg[np.nonzero(self.result.vol_seg  > 0)]).astype('int32')
        self.result.labels      = np.unique(self.result.vol_seg[np.nonzero(self.result.vol_seg != 0)]).astype('int32')

        print('Finished segmenting volume')

        # and to recover our domain boundaries
        seg_gx, seg_gy, seg_gz = np.gradient(self.result.vol_seg, axis=(0,1,2))
        self.result.seg_msk = ((seg_gx**2 + seg_gy**2 + seg_gz**2) == 0)
        del seg_gx, seg_gy, seg_gz
        print('Sorting open regions by flux sign')
        opos_labels=[]
        for i in np.arange(self.result.open_labels.size):
            ri = self.result.open_labels[i]
            n_pos = np.sum(self.source.brr[...,-1][np.nonzero(self.result.vol_seg[...,-1]==ri)]>0) # positive area
            n_neg = np.sum(self.source.brr[...,-1][np.nonzero(self.result.vol_seg[...,-1]==ri)]<0) # negative area
            if (n_pos > 0) | (n_neg > 0):
                if ((n_pos - n_neg)/(n_pos + n_neg)) > 0: opos_labels.append(ri)
            else:
                print('Region ',ri,' has no boundary footprint')
        
        self.result.opos_labels=np.array(opos_labels)

        print('Success \n%%%%%%%%%%%%%%%%%%%%%%%%%%%%')





        ########################
        # end of method ########
        ########################


        

    def determine_adjacency(self):

        # this method determines which HQV subvolumes are associated with each domain
        
        import numpy as np
        from scipy import ndimage as ndi

        if not self.inputs.adj_thrsh: self.inputs.adj_thrsh=0.5
        
        # let's get a mask set up with all of the regions along an axis.
        # It will be a boolean of whether any given pixel is
        # nearer to that region than the threshold

        nph, nth, nrr, nll = self.inputs.nph, self.inputs.nth, self.inputs.nrr, self.result.labels.size
        self.result.adj_msk = np.zeros((nph, nth, nrr, nll), dtype='bool')

        # here we calculate distance transforms
        print('Comparing region specific EDT to hqv_width')
        for i in np.arange(nll):
            #print('Finding distance to region '+str(self.result.labels[i]))
            dist_i = ndi.distance_transform_edt( self.result.vol_seg != self.result.labels[i] ).astype('float32')
            self.result.adj_msk[...,i] = ( dist_i <= ( np.float32(self.inputs.adj_thrsh) * self.result.hqv_width * self.source.crr / self.source.crr.mean() ) )
            print('Finished with region',self.result.labels[i])
        print('Adjacency mask built')

        print('Success              \n%%%%%%%%%%%%%%%%%%%%%%%%%%%%')

        ########################
        # end of method ########
        ########################
            

            

    def get_reg_hqv(self, labels=None, logic=None):
        
        ###########################################################################################
        # this method extracts the hqv associated with a group of domains given a specified logic #
        ###########################################################################################
        
        import numpy as np

        if self.result.adj_msk is None:
            print('adjacency mask not available')
            return

        if labels is None:
             print('must supply label (or list of labels)')
             return
        else:
            try:
                labels=list(labels)
            except TypeError:
                if np.size(labels)==1: 
                    labels=[labels]
                else: 
                    print('labels must be int or iterable type')
                    return 

        if not logic:
            logic='Union'

        # note that the adjacency mask runs over non-zero entries to the label list.
        msk = (np.ones(self.result.hqv_msk.shape, dtype='bool') * (logic=='Intrs'))
        all_labels=self.result.labels
        for i in range(all_labels.size):
            if all_labels[i] in labels:
                if logic=='Union':
                    msk = msk | self.result.adj_msk[...,i] # unions
                elif logic=='Intrs':
                    msk = msk & self.result.adj_msk[...,i] # intersections
                else: print('hqv_type must be one of "Intrs" or "Union"')
        msk = msk & self.result.hqv_msk # project against hqv mask

        return msk


        

        ########################
        # end of method ########
        ########################


    def get_groups(self, data=None, labels=None, group_name=None):
        
        ###################################################################################################
        # this method builds a branch structure showing the various groups with non-trivial intersections #
        ###################################################################################################

        import numpy as np

        if labels is None:
            labels=self.result.labels
            if not group_name: group_name='all_groups'
        elif labels=='all':
            labels=self.result.labels
            if not group_name: group_name='all_groups'
        elif labels=='open':
            labels=self.result.open_labels
            if not group_name: group_name='open_groups'
        elif labels=='clsd':
            labels=self.result.clsd_labels
            if not group_name: group_name='clsd_groups'
        elif labels=='opos':
            labels=self.result.opos_labels
            if not group_name: group_name='opos_groups'
        elif labels=='oneg':
            labels=[el for el in self.result.open_labels if el not in self.result.opos_labels]
            if not group_name: group_name='oneg_groups'
        elif type(labels)==list:
            if not group_name: group_name='custom_group'

        def group_type(labels=None):
            all_open = all([(label in self.result.open_labels)     for label in labels])
            all_clsd = all([(label in self.result.clsd_labels)     for label in labels])
            all_opos = all([(label in self.result.opos_labels)     for label in labels])
            all_oneg = all([(label not in self.result.opos_labels) for label in labels])
            if all_clsd:       iface_type='clsd_mxd'
            elif all_open:
                if all_opos:   iface_type='open_pos'
                elif all_oneg: iface_type='open_neg'
                else:          iface_type='open_mxd'
            else:              iface_type='OCB'
            return iface_type

        group_list = []

        # first we append pairwise groupings
        print('Determining overlapping regions pairs')
        for l1 in labels:
            for l2 in labels[np.nonzero(labels>l1)]:
                group_labels = [l1,l2]
                hqv = self.get_reg_hqv(labels=group_labels, logic='Intrs')
                top = (np.sum(hqv[...,-1]) > 0)
                vol = np.int32(np.sum(hqv))
                if vol > 0:
                    iface_type = group_type(labels=group_labels)
                    #print('overlap found for labels (',group_labels,', vol: ',vol,', top: ',top,', iface: ',iface_type,')')
                    group_obj = Foo()
                    setattr(group_obj, 'labels', group_labels)
                    setattr(group_obj, 'volume', vol)
                    setattr(group_obj, 'tree',   None)
                    setattr(group_obj, 'top',    top)
                    setattr(group_obj, 'iface',  iface_type)
                    group_list.append(group_obj)
                
        # now we explore depth with recursion

        print('Determining higher order overlap groups')
        depth=0
        new_groups=True # initialize number of groups to be explored.
        while new_groups:
            depth+=1
            print('Recursion level:',depth)
            new_groups = False
            label_len_list = [len(grp.labels) for grp in group_list]
            for i in range(len(group_list)):
                if not group_list[i].tree: # need to be explored
                    group_list[i].tree = 'leaf' # leaf unless supporting smaller leaves -- then branch.
                    subgroup_labels = group_list[i].labels
                    n_labels = len(subgroup_labels)
                    ss_same_len_list, = np.nonzero(np.array(label_len_list)==n_labels) # index of groups with same number of antries as current group
                    for j in labels[np.nonzero(labels > max(subgroup_labels))]: # next label to add to group
                        # first we make sure that j has nonempty overlaps with the elements of the group
                        nonempty_count = 0
                        for k in range(n_labels):
                            test_labels = subgroup_labels.copy()
                            test_labels[k] = j # this is the branch group with the new index swapped in for one of the elements
                            test_labels = sorted(test_labels)
                            for ss_same_len in ss_same_len_list: # index of groups with same number of entries as current branch
                                if group_list[ss_same_len].labels == test_labels:
                                    nonempty_count+=1 # this swap works.
                        # from above, there should be as many nonempties as entries (i.e. a&b&c iff a&b & a&c & b&c)
                        if nonempty_count == n_labels: # require that every subgroup had a nonzero entry.        
                            supergroup_labels = [l for k in [subgroup_labels, [j]] for l in k]
                            hqv = self.get_reg_hqv(labels=supergroup_labels, logic='Intrs')
                            top = (np.sum(hqv[...,-1]) > 0)
                            vol = np.int32(np.sum(hqv))
                            if vol > 0:
                                iface_type = group_type(labels=supergroup_labels)
                                #print('overlap found for labels (',supergroup_labels,', vol: ',vol,', top: ',top,', iface: ',iface_type,')')
                                group_obj = Foo()
                                setattr(group_obj, 'labels', supergroup_labels)
                                setattr(group_obj, 'volume', vol)
                                setattr(group_obj, 'tree',   None)
                                setattr(group_obj, 'top',    top)
                                setattr(group_obj, 'iface',  iface_type)
                                group_list.append(group_obj)
                                group_list[i].tree = 'branch' # leaf found, so group becomes branch.
                                new_groups = True

        print('Overlap group search complete')
        # now we'll add the result to the list of groups.
        if not self.result.groupings: self.result.groupings=Foo()
        if hasattr(self.result.groupings, group_name):
            print('Overwriting previous entry for group: ',group_name)
        else:
            print('Populating new entry for group: ',group_name)
        setattr(self.result.groupings, group_name, group_list)
        
        print('Success \n%%%%%%%%%%%%%%%%%%%%%%%%%%%%')

        ########################
        # end of method ########
        ########################




    




    def make_null_box(self, null_num=None, box_size=None):

        ####################################################################################################
        # this method generates a group of indices describing the corners of a box containing a given null #
        ####################################################################################################

        import numpy as np
        
        if not null_num: null_num=0
        if not box_size: box_size=2
        if ((box_size%2)>0):
            box_size+=1
            print('box must be of even size -- increasing by one')

        crr = self.source.crr[0,0,:]
        cth = self.source.cth[0,:,0]
        cph = self.source.cph[:,0,0]
        nrr = np.float32(crr.size)
        nth = np.float32(cth.size)
        nph = np.float32(cph.size)

        null_locs=self.source.null_locs.T

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

        ########################
        # end of method ########
        ########################




    def get_null_reg_dist(self, labels=None, null_list=None):

        #################################################################################################
        # this method gets the distnaces from each null to each domain, in the form of a pair of arrays #
        #################################################################################################

        import numpy as np

        if labels is None:
            print('region list not supplied -- selecting all')
            labels = self.result.labels
            labelname='all_labels'

        null_locs = self.source.null_locs
        N_null=null_locs.shape[1]

        if null_list is None:
            print('null list not supplied -- selecting all')
            null_list=np.arange(N_null)
            nullname='all_nulls'
        
        int_distance = np.zeros((np.size(labels), np.size(null_list))).astype('float32')
        hqv_distance = np.zeros((np.size(labels), np.size(null_list))).astype('float32')

        for j in range(np.size(null_list)):
            n = null_list[j]
            box = np.array(self.make_null_box(null_num=n, box_size=2)).T
            box_indices = (box[0], box[1], box[2])
            # we'll do measurements in sphericals but assuming orthonormality -- this is innacurate for large distances but we only care about small distances.
            # dividing by the metrics locally puts the distances into pixels, for easy comparison to the hqv_width, also in pixels.
            dist_rr = (self.source.crr - self.inputs.solrad*self.source.null_locs.T[n,2]) / self.source.metrr
            dist_th = self.source.crr*(np.pi / 180)*(self.source.cth - self.source.null_locs.T[n,1]) / self.source.metth 
            dist_ph = self.source.crr*np.cos(self.source.cth*np.pi/180)*(np.pi/180) * (self.source.cph - self.source.null_locs.T[n,0]) / self.source.metph
            dist_mm = np.sqrt(dist_rr**2 + dist_th**2 + dist_ph**2)
            del dist_rr, dist_th, dist_ph
            print('finding distances to null:',n)
            for i in range(np.size(labels)):
                # have to get the mask for the region hqv
                hqv_msk = self.get_reg_hqv(labels=labels[i])
                # test the typical hqv mask at the null box
                hqv_box_dist = np.mean(~hqv_msk[box_indices])
                if (hqv_box_dist < 0.5): # mostly inside domain.
                    hqv_distance[i,j] = hqv_box_dist
                else:
                    hqv_distance[i,j] = max(0.5, dist_mm[np.nonzero(hqv_msk)].min())
                int_msk = (self.result.vol_seg == labels[i])
                # test the typical int msk at the null box
                int_box_dist = np.mean(~int_msk[box_indices])
                if (int_box_dist < 0.5):
                    int_distance[i,j] = int_box_dist
                else:
                    int_distance[i,j] = max(0.5, dist_mm[np.nonzero(int_msk)].min())
                # with these conventions, the distance can go to zero, even if the null is not centered on a given pixel.
                #print('region: '+str(labels[i])+', null: '+str(n)+', hqv distance: '+str(hqv_distance[i,j])+', int distance:'+str(int_distance[i,j]))

        dist_object = Foo()
        setattr(dist_object, 'regions', np.int32(labels))
        setattr(dist_object, 'nulls', np.int32(null_list))
        setattr(dist_object, 'int_distance', np.float32(int_distance))
        setattr(dist_object, 'hqv_distance', np.float32(hqv_distance))

        if (labelname == 'all_labels') and (nullname == 'all_nulls'):
            print('Populating exhaustive list')
            self.result.null_to_reg_dist = dist_object
        else: return dist_object
        
        print('Success \n%%%%%%%%%%%%%%%%%%%%%%%%%%%%')

        


        ########################
        # end of method ########
        ########################


    def get_nll_regs(null_list=None):

        ###################################################################################
        # this method gets the domains associated with a given null (not used at present) #
        ###################################################################################
        
        if null_list is None:
            print('must supply null indices')
            return
        
        labels=self.result.labels
        null_regs = []
        for i in range(labels.size):
            for j in null_list:
                jbox=self.make_null_box(null_num=j, box_size=2)
                jmsk = self.result.adj_msk[...,i]
                if np.sum([jmsk[coords] for coords in jbox]) > 0:
                    null_regs.append((labels[i], j))

        return null_regs

        ########################
        # end of method ########
        ########################


    def get_adj_nll(labels=None, logic=None):

        import numpy as np
        
        ###############################################################################
        # this method gets the nulls adjacent to a given domain (not used at present) #
        ###############################################################################
        
        hqv_msk = self.get_reg_hqv(labels=labels, logic=logic)

        # get nulls
        null_locs=self.result.null_locs.T
        null_locs=null_locs[np.argsort(null_locs[:,2]),:]
        N_null=null_locs.shape[0]

        # find overlap
        adj_nll_list=[]
        for n in range(N_null):
            nll_box = self.make_null_box(null_num=n, box_size=2)
            if np.sum(hqv_msk[coords] for coords in nll_box) > 0:
                adj_nll_list.append(labels, logic, n)

        return adj_nll_list

        ########################
        # end of method ########
        ########################






















        


    #########################################
    # model methods for data I/O ############
    #########################################
                
    def save_data(self, fname=None, bitpack=True):
        import numpy
        
        if not fname:
            if not self.inputs.q_dir:
                base = './'
            else:
                base = self.inputs.q_dir
            fname=base+'/squash_mod.pickle'

        def pickle_dump(obj, file_path):
            import pickle
            with open(file_path, "wb") as f:
                return pickle.dump(obj, Bytefile(f), protocol=pickle.HIGHEST_PROTOCOL)

        
        if self.result.adj_msk is None:
            pass
        else:
            if bitpack: # this is to put the adjacency boolean array at a reasonable size on disk
                print('Packing Adj mask to bitwise array for efficient storage on disk')
                self.result.adj_msk_shape = self.result.adj_msk.shape
                self.result.adj_msk_boolsize = self.result.adj_msk.size
                temp = self.result.adj_msk.copy()
                self.result.adj_msk = numpy.packbits(temp, axis=None)
            
        pickle_dump(self, fname)
        
        print('Model written to '+str(fname))

        if self.result.adj_msk is None:
            pass
        else:
            if bitpack:
                self.result.adj_msk=temp
                del temp
            
    def load_data(self, fname=None, bitpack=True):
        import numpy
        
        if not fname:
            if not self.inputs.q_dir:
                base = './'
            else:
                base = self.inputs.q_dir
            fname=base+'/squash_mod.pickle'
            
        def pickle_load(file_path):
            import pickle
            with open(file_path, "rb") as f:
                return pickle.load(Bytefile(f))
            
        model = pickle_load(fname)
        
        ### iterate over restored contents and populate model attributes.
        for key in model.__dict__.keys():
            if key[0] != '_':
                setattr(self, key, getattr(model, key))
            else:
                print('skipping attribute',key)
        print('Model keys: '+str([key for key in model.__dict__.keys()])+ ' read from '+str(fname))

        if self.result.adj_msk is None:
            pass
        else:
            if self.result.adj_msk_boolsize is not None and bitpack:
                print('Unpacking Adj mask due to bitwise storage on disk')
                self.result.adj_msk = numpy.array((numpy.unpackbits(self.result.adj_msk)[:self.result.adj_msk_boolsize]).reshape(self.result.adj_msk_shape), dtype='bool')

        return self
    
    def export_vtk(self, fname=None, rr_rng=None, th_rng=None, ph_rng=None):
        pass
    



    
    ###############
    # visualization
    ###############

    def gauss_slice(dcube, axis, center=0.5, sigma=0.1):
        # just a gaussian filter for use in LOS integration #
        import numpy as np
        
        nx,ny,nz = dcube.shape
        tmpx = np.linspace(0, 1, nx)
        tmpy = np.linspace(0, 1, ny)
        tmpz = np.linspace(0, 1, nz)

        if axis==0: tmp = np.reshape(np.roll(np.exp(- 0.5 * ((tmpx-0.5) / sigma)**2), int((center - 0.5)*nx) ), [tmpx.size,1,1])
        if axis==1: tmp = np.reshape(np.roll(np.exp(- 0.5 * ((tmpy-0.5) / sigma)**2), int((center - 0.5)*ny) ), [1,tmpy.size,1])
        if axis==3: tmp = np.reshape(np.roll(np.exp(- 0.5 * ((tmpz-0.5) / sigma)**2), int((center - 0.5)*nz) ), [1,1,tmpz.size])

        return np.mean(tmp*cube, axis=axis)/np.mean(tmp, axis=axis)
    
    def segcmap(self, shuffle_colors=None):

        import numpy as np
        import pylab as plt
        import matplotlib as mpl
        
        # define a custom color for segmentation visualization
        labels = self.result.labels
        openlin = np.linspace(0, 1, np.size(labels[np.argwhere(labels < 0)]) + 1)[1:]
        clsdlin = np.linspace(0, 1, np.size(labels[np.argwhere(labels > 0)]) + 1)[1:]
        if shuffle_colors:
            from random import shuffle
            shuffle(openlin)
            shuffle(clsdlin)
        opencolors = plt.cm.winter(openlin)
        clsdcolors = plt.cm.autumn(clsdlin)
        whitecolor = plt.cm.binary(0)
        blackcolor = plt.cm.binary(255)
        mycolors = np.vstack((np.flip(opencolors, axis=0),blackcolor,clsdcolors))
        cmap = mpl.colors.LinearSegmentedColormap.from_list('my_colormap', mycolors)

        return cmap
    
    def visualize(self, window=None, figsize=None):

        import numpy as np
        import pylab as plt
        import matplotlib as mpl
        from matplotlib.widgets import Slider, RadioButtons, Button

        rr=None
        th=None
        ph=None
        
        P_null=self.source.null_locs.T
        P_null=P_null[np.argsort(P_null[:,2]),:]
        N_null=P_null.shape[0]

        # get coordinates
        crr = self.source.crr[0,0,:]/696.
        cth = self.source.cth[0,:,0]
        cph = self.source.cph[:,0,0]
        nrr = crr.size
        nth = cth.size
        nph = cph.size
        th_min= cth.min()
        th_max= cth.max()
        ph_min= cph.min()
        ph_max= cph.max()
        rr_min= crr.min()
        rr_max= crr.max()

        if not window: window=0

        if not figsize: figsize=(16,9)
        elif type(figsize)!=tuple:
            print('figsize must be a tuple')

        if 'fig' not in locals():
            fig = plt.figure(window, figsize=figsize)
        fig.clf()

        if rr is None: rr=crr[-1]
        if th is None: th=cth[int(nth/2)]
        if ph is None: ph=cph[int(nph/2)]
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

        dcube = self.source.slog10q
        dmask = np.ones(dcube.shape, dtype='bool')
        ctable ='viridis'
        vrange = (-4,4)

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

        mask_key=None
        data_key='slog10q'
        inv_mask=False
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

            if hasattr(self.source, data_key):
                dcube=getattr(self.source, data_key)
            if hasattr(self.result, data_key):
                dcube=getattr(self.result, data_key)
            if data_key=='brr':
                ctable='bwr_r'
                brms= np.around(np.sqrt((self.source.brr[...,irr]**2).mean()), decimals=2)
                vrange = (-brms, brms)
            elif data_key=='slog10q':
                ctable='viridis'
                vrange = (-4,4)
            elif data_key=='vol_seg':
                ctable = self.segcmap()
                vrange = (dcube.min(), dcube.max())
            else:
                print('unknown data key')
                return
            
            if not mask_key: dmask=np.ones(dcube.shape, dtype='bool')
            elif mask_key=='hqv_msk':
                if inv_mask: dmask = self.result.hqv_msk
                else: dmask = ~self.result.hqv_msk
            elif mask_key=='seg_msk':
                if inv_mask: dmask = ~self.result.seg_msk
                else: dmask = self.result.seg_msk

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
            if null_vis:
                null_label=axbt.text(0,0.5,r'$N_i=$'+str(c_null))
            else:
                null_label=axbt.text(0,0.5,'')



            if data_key == 'vol_seg':
                tickvals = [dcube.min(), 0, dcube.max()]
                ticknames = ['open', 'OCB', 'closed']
            else:
                tickvals = np.linspace(vrange[0], vrange[1], 5)
                ticknames = ['{:2.2f}'.format(i) for i in tickvals]

            cbar.set_ticks(tickvals)
            cbar.ax.set_yticklabels(ticknames)

            return 0

        def update_rr_coord(val):
            nonlocal draw_params, null_vis, null_ini, rr_nodraw
            draw_params['rr'] = srr.val
            if (not rr_nodraw): # behavior when resetting a single coordinate allows immediate redraw and logic
                if (null_vis) & (not null_ini): null_vis=False
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
            if (not th_nodraw):
                if (null_vis) & (not null_ini): null_vis=False
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
            if (not ph_nodraw):
                if (null_vis) & (not null_ini): null_vis=False
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
            if label =='HQV mask':  draw_params['mask_key']='hqv_msk'
            if label =='seg bound': draw_params['mask_key']='seg_msk'
            if label =='No mask':   draw_params['mask_key']=None
            #print(mask_key)
            redraw()
            fig.canvas.draw_idle()

        def update_data_key(label):
            nonlocal draw_params
            if label=='slog10 Q':     draw_params['data_key']='slog10q'
            if label=='seg map':      draw_params['data_key']='vol_seg'
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
            if (not null_vis): null_vis=True
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
            if (not null_vis): null_vis=True
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


###################################################
# a method the larger module for cloaning objects #

###################################################

def portattr(target_object, name, attribute):
    # this routine works like setattr but it detects model objects and instantiates from new
    namelist = ['Inputs', 'Source', 'Result', 'Foo']
    classname = attribute.__class__.__name__
    if classname in namelist: # attribute is a class object unto itself
        print('Attribute', name, 'is of type', classname, 'and must be set dynamically')
        try:
            setattr(target_object, name, object.__new__(type(attribute))) # dynamically instantiate new instance
            for key in attribute.__dict__.keys():
                if key[0] != '_':
                    print('Setting subattribute', key)
                    portattr(getattr(target_object, name), key, getattr(attribute, key))
        except:
            print('Cannot instantiate',name)
    else: # attribute is simply object
        print('Attribute', name, 'is not an HQVseg object. Attempting to set statically')
        try:
            setattr(target_object, name, attribute)
            print('Attribute', name, 'set statically.')
        except AttributeError:
            print('Attribute', name, 'cannot be set.')


