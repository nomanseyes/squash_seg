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
                 auto_imp=None, auto_seg=None, protocol=None, vis_title=None):

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
        self.vis_title=vis_title

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
                 vol_seg=None, seg_msk=None, adj_msk=None, intersections=None, detached_HQVs=None, \
                 null_to_region_dist=None, null_to_detached_dist=None, \
                 open_labels=None, clsd_labels=None, opos_labels=None, oneg_labels=None, labels=None):

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
        self.oneg_labels        =oneg_labels

        # these come from post-processing on the various domain connectivities

        self.adj_msk               = adj_msk
        self.adj_msk_shape         = None
        self.adj_msk_boolsize      = None
        self.intersections         = intersections
        self.detached_HQVs         = detached_HQVs
        self.null_to_region_dist   = null_to_region_dist
        self.null_to_detached_dist = null_to_detached_dist


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
        self.find_intersections()
        self.find_detached_HQVs()
        self.get_null_reg_dist()
        self.get_null_dtch_dist()

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
        return self

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

        import os
        import numpy as np
        import pandas as pd

        # allows generalization for other data formats
        if not self.inputs.protocol: self.inputs.protocol='standard_QSLsquasher'

        # import assuming typical data format
        if self.inputs.protocol=='standard_QSLsquasher':
            if self.inputs.q_dir is None:
                self.inputs.q_dir = os.getcwd()
            print('Loading Q data')
            self.source.slog10q=(np.array(pd.read_table(self.inputs.q_dir+'/grid3d.dat',header=None).astype('float32')))[...,0].reshape((self.inputs.nph,self.inputs.nth,self.inputs.nrr))
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
            if self.inputs.b_dir is None:
                if self.inputs.q_dir is None:
                    self.inputs.b_dir = os.getcwd()+'/bfield_data/'
                else:
                    self.inputs.b_dir = self.inputs.q_dir+'/bfield_data/'
            print('Loading B data')
            # these are the native coordinates of the magnetic field grid
            ph_mag = np.array(pd.read_table(self.inputs.b_dir+'/xs0.dat',header=None))[...,0].astype('float32')
            th_mag = np.array(pd.read_table(self.inputs.b_dir+'/ys0.dat',header=None))[...,0].astype('float32')
            rr_mag = np.array(pd.read_table(self.inputs.b_dir+'/zs0.dat',header=None))[...,0].astype('float32') * self.inputs.solrad
            nx=ph_mag.size
            ny=th_mag.size
            nz=rr_mag.size
            # these are the components of the magnetic field on the native grid
            # minus in b_theta is due to Griffiths' def
            b_ph_nat =   (np.array(pd.read_table(self.inputs.b_dir+'/bx0.dat',header=None))[...,0].reshape((nz,ny,nx)) ).transpose((2,1,0)).astype('float32')
            b_th_nat = - (np.array(pd.read_table(self.inputs.b_dir+'/by0.dat',header=None))[...,0].reshape((nz,ny,nx)) ).transpose((2,1,0)).astype('float32')
            b_rr_nat =   (np.array(pd.read_table(self.inputs.b_dir+'/bz0.dat',header=None))[...,0].reshape((nz,ny,nx)) ).transpose((2,1,0)).astype('float32')
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
            null_loc_fname = self.inputs.b_dir+'/nullpositions.dat'
            if os.path.isfile(null_loc_fname):
                from scipy.io import readsav
                null_dict=readsav(null_loc_fname, python_dict=True)
                self.source.null_locs=null_dict['nulls'].astype('float32')
                P_null=self.source.null_locs.T
                P_null=P_null[np.argsort(P_null[:,2]),:]
                self.source.null_locs = P_null.T
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
            print('LTQ threshold specified directly as: ', ltq_thrsh)
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

        hqv_msk = self.result.hqv_msk
        slog10q = self.source.slog10q
        crr     = self.source.crr
        brr     = self.source.brr
        GlnQp   = self.result.GlnQp

        # for global models, we have to blow out the dimensions in phi to move the boundary away
        if self.inputs.glbl_mdl is None: self.inputs.glbl_mdl=True
        
        if self.inputs.glbl_mdl:
            print('expanding to double-size domain')
            hqv_msk = self.global_expand(hqv_msk)
            slog10q = self.global_expand(slog10q)
            crr     = self.global_expand(crr)
            brr     = self.global_expand(brr)
            GlnQp   = self.global_expand(GlnQp)
            #(self.result.hqv_msk, self.source.slog10q, self.source.crr, self.source.brr, self.result.GlnQp) = (None, None, None, None, None)


        vol_seg = np.zeros(hqv_msk.shape, dtype='int32')### initiate segmentation label array.
        # it's useful to have masks for open and closed flux. We consider undecided flux to be open.
        opn_msk = (slog10q < 0)
        if self.inputs.ss_eof: opn_msk[...,-1]=True
        cls_msk = (slog10q > 0)

        print('Calculating distance transforms')

        # it's also useful to have the distance from the interior of the hqv regions to their boundary, and the other way.
        # now we do our distance transform and masking
        hqv_dist = ndi.distance_transform_edt(~hqv_msk).astype('float32') # distance to nearest hqv
        reg_width = np.mean(4.*hqv_dist[np.nonzero(~hqv_msk)]).astype('float32') # full width is 4 times average distance to boundary
        print('Vol width: ',reg_width)
        reg_dist = ndi.distance_transform_edt(hqv_msk).astype('float32') # distance to nearest low q region
        hqv_width = np.mean(4.*reg_dist[np.nonzero(hqv_msk)]).astype('float32') # full width is 4 times average distance to boundary
        print('HQV width: ',hqv_width)
        del(reg_dist)

        print('Performing discrete flux labeling')

        if not self.inputs.pad_ratio: self.inputs.pad_ratio=0.25 # default is pad out to a typical quarter-width.

        # now we'll label the open flux domains, above min height.
        # we'll pad the hqv mask by a distance proportionate to the hqv_halfwidth, including a radial scaling to accomodate thicker hqvs at larger radii
        pad_msk = ( hqv_dist > self.inputs.pad_ratio * hqv_width * (crr / crr.mean()) ) # empirical radial dependence...
        open_label_mask = pad_msk & (slog10q < 0) & (crr >= self.inputs.bot_rad * self.inputs.solrad)
        clsd_label_mask = pad_msk & (slog10q > 0) & (crr >= self.inputs.bot_rad * self.inputs.solrad)
        vol_seg = np.zeros(pad_msk.shape, dtype='int32')
        vol_seg -= skim.label(open_label_mask).astype('int32') # all pixels not within or adjacent to a hqv
        vol_seg += skim.label(clsd_label_mask).astype('int32') # all pixels not within or adjacent to a hqv
        del(open_label_mask, clsd_label_mask)
            
        print('Removing domains with sub-minimum volume')
        
        crr_mean = crr.mean()
        hqv_vol = (0.5*hqv_width)**3
        
        for reg in np.unique(vol_seg[np.nonzero(vol_seg)]):
            reg_ss = np.nonzero(vol_seg == reg)
            # threshold size for valid regions -- threshold increases quadratically with height to allow smaller closed domains
            # looks a bit odd. It's just sum over reg_ss.size, weighted by radial scaling, compared to hqv_vol. 
            if np.sum(crr_mean**2 / crr[reg_ss]**2) < hqv_vol: 
                vol_seg[reg_ss] = 0  # zero in mask, unchanged else.
        del(reg_ss, crr_mean, hqv_vol)

        # and we'll define the label groups for persistent open and closed domains
        clsd_labels = np.unique(vol_seg[np.nonzero(vol_seg > 0)])
        open_labels = np.unique(vol_seg[np.nonzero(vol_seg < 0)])
        opos_labels=[]
        oneg_labels=[]
        for i in np.arange(open_labels.size):
            ri = open_labels[i]
            n_pos = np.sum(brr[...,-1][np.nonzero(vol_seg[...,-1]==ri)]>0) # positive area
            n_neg = np.sum(brr[...,-1][np.nonzero(vol_seg[...,-1]==ri)]<0) # negative area
            if (n_pos > 0) | (n_neg > 0):
                if ((n_pos - n_neg)/(n_pos + n_neg)) > 0: opos_labels.append(ri)
                if ((n_pos - n_neg)/(n_pos + n_neg)) < 0: oneg_labels.append(ri)
        opos_labels=np.array(opos_labels)
        oneg_labels=np.array(oneg_labels)

        print('Performing watershed backfill into HQV padding')
        absQp = 10**np.clip(np.absolute(slog10q), np.log10(2), 10).astype('float32')
        log10_SNQ =  np.log10(absQp + (crr / crr.max()) * GlnQp**2).astype('float32')
        del(absQp)

        # Initial watershed phase
        # First we grow regions into the padding layer outside the hqv mask
        stime = time.time()
        vol_seg += (mor.watershed(log10_SNQ, vol_seg * opn_msk, mask=(opn_msk & ~hqv_msk), watershed_line=False) * ((vol_seg==0) & opn_msk & ~hqv_msk)).astype('int32')
        print('Open flux backfill completed in '+str(int(time.time()-stime))+' seconds')
        stime = time.time()
        vol_seg += (mor.watershed(log10_SNQ, vol_seg * cls_msk, mask=(cls_msk & ~hqv_msk), watershed_line=False) * ((vol_seg==0) & cls_msk & ~hqv_msk)).astype('int32')
        print('Clsd flux backfill completed in '+str(int(time.time()-stime))+' seconds')

        print('Enforcing boundary connectivity')
        # And we require that all regions are associated with a boundary
        # These loops are split up to allow for different definitions between open and closed domains
        for reg in open_labels:
            tmp_msk = (vol_seg == reg)
            if np.sum(tmp_msk[...,-1]) == 0: # open domains must intersect the top boundary
                vol_seg = vol_seg * ~tmp_msk # zero in mask, unchanged else.
        for reg in clsd_labels:
            tmp_msk = (vol_seg == reg)
            if np.sum(tmp_msk[...,0]) == 0: # closed domains must intersection the bottom boundary.
                vol_seg = vol_seg * ~tmp_msk # zero in mask, unchanged else.
                del(tmp_msk)
                


        print('Performing restricted watershed backfill into HQV mask')
        # Second, we grow regions using the same-type mask, which just expands open-open, close-close.
        stime = time.time()
        vol_seg += (mor.watershed(log10_SNQ, vol_seg * opn_msk, mask=opn_msk, watershed_line=False) * ((vol_seg==0) & opn_msk)).astype('int32')
        print('Open flux backfill completed in '+str(int(time.time()-stime))+' seconds')
        stime = time.time()
        vol_seg += (mor.watershed(log10_SNQ, vol_seg * cls_msk, mask=cls_msk, watershed_line=False) * ((vol_seg==0) & cls_msk)).astype('int32')
        print('Clsd flux backfill completed in '+str(int(time.time()-stime))+' seconds')



        print('Performing transparent watershed backfill into HQV mask')
        # Third, we grow regions through opposite type, but only within a hqv, where type mixing is expected.
        stime = time.time()
        vol_seg += (mor.watershed(log10_SNQ, vol_seg * opn_msk, mask=hqv_msk, watershed_line=False) * ((vol_seg==0) & opn_msk)).astype('int32')
        print('Open flux backfill completed in '+str(int(time.time()-stime))+' seconds')
        stime = time.time()
        vol_seg += (mor.watershed(log10_SNQ, vol_seg * cls_msk, mask=hqv_msk, watershed_line=False) * ((vol_seg==0) & cls_msk)).astype('int32')
        print('Clsd flux backfill completed in '+str(int(time.time()-stime))+' seconds')



        print('Performing watershed backfill into residual domains')
        # Finally, we grow null regions with no preference, allowing open and closed to compete.
        stime = time.time()
        vol_seg += (mor.watershed(           1/(1 + hqv_dist), vol_seg,                         watershed_line=False) * ((vol_seg==0))).astype('int32')
        print('Final flux backfill completed in '+str(int(time.time()-stime))+' seconds')
        # There may still be unnasigned regions. But these will be outliers buried deep within opposite flux types.
        del(hqv_dist, opn_msk, cls_msk) # don't need these anymore


        # get the pure interface boundaries from the domain map
        seg_gx, seg_gy, seg_gz = np.gradient(vol_seg, axis=(0,1,2))
        seg_msk = ((seg_gx**2 + seg_gy**2 + seg_gz**2) == 0)
        del(seg_gx, seg_gy, seg_gz)

        # Here we enforce periodicity
        if self.inputs.glbl_mdl:
            print('Associating labels across phi=0 boundary')
            print('Closed flux...')
            self.associate_labels(vol_seg, axis=0, label_subset=clsd_labels, mask=seg_msk, use_volume=True, exp_loop=False)
            print('Open positive flux...')
            self.associate_labels(vol_seg, axis=0, label_subset=opos_labels, mask=seg_msk, use_volume=True, exp_loop=False)
            print('Open negative flux...')
            self.associate_labels(vol_seg, axis=0, label_subset=oneg_labels, mask=seg_msk, use_volume=True, exp_loop=False)
            # If global we need to restore the original array shape
            # update interface boundaries from the domain map
            seg_gx, seg_gy, seg_gz = np.gradient(vol_seg, axis=(0,1,2))
            seg_msk = ((seg_gx**2 + seg_gy**2 + seg_gz**2) == 0)
            del(seg_gx, seg_gy, seg_gz)
            print('reducing from double-size domain')
            vol_seg = self.global_reduce(vol_seg)
            pad_msk = self.global_reduce(pad_msk)
            hqv_msk = self.global_reduce(hqv_msk)
            slog10q = self.global_reduce(slog10q)
            crr     = self.global_reduce(crr)
            brr     = self.global_reduce(brr)
            GlnQp   = self.global_reduce(GlnQp)
            seg_msk = self.global_reduce(seg_msk)
        
        print('Relabeling to remove obsolete domains') 
        # now let's relabel with integer labels removing gaps
        # first we need this list of labels that persist
        clsd_labels_old = np.unique(vol_seg[np.nonzero(vol_seg > 0)])
        open_labels_old = np.unique(vol_seg[np.nonzero(vol_seg < 0)])
        opos_labels_old = np.array([label for label in opos_labels if label in open_labels_old])
        oneg_labels_old = np.array([label for label in oneg_labels if label in open_labels_old])

        # now we generate new lists of same size with no gaps
        open_labels = - np.arange(1, open_labels_old.size + 1).astype('int32')
        opos_labels = - np.arange(1, opos_labels_old.size + 1).astype('int32')
        oneg_labels = - np.arange(1, oneg_labels_old.size + 1).astype('int32') + opos_labels.min() # offset by largest negative opos label
        clsd_labels = + np.arange(1, clsd_labels_old.size + 1).astype('int32')

        # now we'll shuffle the ordering of the old ones to randomize the locations
        np.random.seed(opos_labels_old.size) # repeatable random seed
        np.random.shuffle(opos_labels_old) # random shuffle of domain order
        np.random.seed(oneg_labels_old.size) 
        np.random.shuffle(oneg_labels_old) # same for oneg
        np.random.seed(clsd_labels_old.size)
        np.random.shuffle(clsd_labels) # same for clsd

        swapped = np.zeros(vol_seg.shape, dtype='bool') # boolean to track already swapped domains

        for i in np.arange(opos_labels_old.size):
            swap_msk = ((vol_seg == opos_labels_old[i]) & ~swapped)
            swapped  = swapped | swap_msk
            vol_seg[np.nonzero(swap_msk)] = opos_labels[i]
        
        for i in np.arange(oneg_labels_old.size):
            swap_msk = ((vol_seg == oneg_labels_old[i]) & ~swapped)
            swapped  = swapped | swap_msk
            vol_seg[np.nonzero(swap_msk)] = oneg_labels[i]
        
        for i in np.arange(clsd_labels_old.size):
            swap_msk = ((vol_seg == clsd_labels_old[i]) & ~swapped)
            swapped  = swapped | swap_msk
            vol_seg[np.nonzero(swap_msk)] = clsd_labels[i]
            
        del(swapped, swap_msk, opos_labels_old, oneg_labels_old, clsd_labels_old)

        # get the whole label list anew
        labels = np.unique(vol_seg[np.nonzero(vol_seg)])

        print('Finished segmenting volume')

        # and store these permanently
        self.result.reg_width          =reg_width
        self.result.hqv_width          =hqv_width
        self.result.pad_msk            =pad_msk
        self.result.vol_seg            =vol_seg
        self.result.seg_msk            =seg_msk
        self.result.labels             =labels
        self.result.open_labels        =open_labels
        self.result.clsd_labels        =clsd_labels
        self.result.opos_labels        =opos_labels
        self.result.oneg_labels        =oneg_labels
        
        # these should be redundant definitions in the non-global case
        # in the global case, these were set to None to save memory so we restore them here.
        #self.result.hqv_msk            =hqv_msk
        #self.source.slog10q            =slog10q
        #self.source.crr                =crr
        #self.source.brr                =brr
        #self.result.GlnQp              =GlnQp

        

        print('Success \n%%%%%%%%%%%%%%%%%%%%%%%%%%%%')

        return 0



        ########################
        # end of method ########
        ########################







        
    def global_expand(self, input_array, in_place=None):
        import numpy as np
        if in_place is None: in_place = True
        if in_place:
            array = input_array
        else:
            array = input_array.copy()
            
        if array.shape[0] == self.inputs.nph:
            array = np.roll(np.concatenate((array, array), axis=0), np.int(self.inputs.nph/2), axis=0)
            return array
        else:
            print('Object dimensions must match global coordinate dimensions')
            return -1

        ########################
        # end of method ########
        ########################




        
    def global_reduce(self, input_array, in_place=None):
        import numpy as np
        if in_place is None: in_place = True
        if in_place:
            array = input_array
        else:
            array = input_array.copy()
        if array.shape[0] == 2*self.inputs.nph:
            array = np.roll(array, -np.int(self.inputs.nph/2), axis=0)[0:self.inputs.nph,...]
            return array
        else:
            print('Object dimensions must match expanded global coordinate dimensions')
            return -1

        ########################
        # end of method ########
        ########################






    def associate_labels(self, input_array, axis=None, use_boundary=None, use_volume=None, lft_index_range=None, rgt_index_range=None, label_subset=None, mask=None, in_place=None, return_pairs=None, exp_loop=None):

        # this routine associates discrete domain labels across within the interior of the volume
        # first we get the slices at the boundary of interest in the form of an index array
        import time
        import pdb
        import numpy as np

        if in_place is None: in_place = True
        if return_pairs is None: return_pairs = False
        if mask is None: mask=np.ones(input_array.shape, dtype='bool')
        if axis is None: axis=0
        if (lft_index_range is None) and (rgt_index_range is None):
            if (use_volume is None) and (use_boundary is None):
                use_boundary=True

        if in_place:
            label_array = input_array
        else:
            label_array = input_array.copy()

        mp_ss = np.int32(label_array.shape[axis]/2)

        stime = time.time()
        
        if axis==0:
            if use_boundary:
                lft_array = (label_array*mask)[0 , :, :]
                rgt_array = (label_array*mask)[-1, :, :]
            elif use_volume:
                lft_array = (label_array*mask)[:mp_ss, :, :]
                rgt_array = (label_array*mask)[mp_ss:, :, :]
            else:
                lft_array = (label_array*mask)[lft_index_range[0]:lft_index_range[1], :, :]
                rgt_array = (label_array*mask)[rgt_index_range[0]:rgt_index_range[1], :, :]
        elif axis==1:
            if use_boundary:
                lft_array = (label_array*mask)[:, 0 , :]
                rgt_array = (label_array*mask)[:, -1, :]
            elif use_volume:
                lft_array = (label_array*mask)[:, :mp_ss, :]
                rgt_array = (label_array*mask)[:, mp_ss:, :]
            else:
                lft_array = (label_array*mask)[:, lft_index_range[0]:lft_index_range[1], :]
                rgt_array = (label_array*mask)[:, rgt_index_range[0]:rgt_index_range[1], :]
        elif axis==2:
            if use_boundary:
                lft_array = (label_array*mask)[:, :,  0]
                rgt_array = (label_array*mask)[:, :, -1]
            elif use_volume:
                lft_array = (label_array*mask)[:, :, :mp_ss]
                rgt_array = (label_array*mask)[:, :, mp_ss:]
            else:
                lft_array = (label_array*mask)[:, :, lft_index_range[0]:lft_index_range[1]]
                rgt_array = (label_array*mask)[:, :, rgt_index_range[0]:rgt_index_range[1]]
        else: pass

        # now we get the list of labels at the boundary, excluding zero entries.
        lft_labels = np.array(np.unique(lft_array[np.nonzero(lft_array)]))
        rgt_labels = np.array(np.unique(rgt_array[np.nonzero(rgt_array)]))

        # we'll only consider labels in the reduced group if specified
        if label_subset is None:
            label_subset=np.sort(np.unique(list(set(lft_labels).union(set(rgt_labels)))))
        elif type(label_subset) is type(list(())):
            label_subset = np.array(label_subset)
        elif type(label_subset) is type(np.array(())):
            pass
        else:
            print('supplied label group must be a list or an array')
            return -1
        
        lft_labels = [label for label in lft_labels if label in label_subset]
        rgt_labels = [label for label in rgt_labels if label in label_subset]
        
        # and we set up a table to keep track of how these labels associate with each other
        # row and column both must contain all entries, so the swap table will be build on the label_subset group
        pair_map = np.zeros((label_subset.size, label_subset.size), dtype='bool')

        # now we loop over these labels and look for cross matches.
        for lft_label in lft_labels:
            lft_idx = np.nonzero(label_subset == lft_label)
            lft_msk = (lft_array == lft_label)
            #print('masking left region:',lft_label)
            if exp_loop is False:
                rgt_matches = np.unique(rgt_array[np.nonzero(lft_msk)])
                #print('found matches in region(s):',rgt_matches)
                for rgt_label in rgt_matches:
                    if (rgt_label != lft_label) and (rgt_label != 0):
                        #print('recording match in region:',rgt_label)
                        rgt_idx = np.nonzero(label_subset == rgt_label)
                        pair_map[lft_idx, rgt_idx]=True
            else:
                for rgt_label in rgt_labels:
                    #print('masking right region:',rgt_label)
                    if (rgt_label != lft_label): # no reason to match against the same region
                        rgt_idx = np.nonzero(label_subset == rgt_label)
                        rgt_msk = (rgt_array == rgt_label)
                        if np.max(lft_msk & rgt_msk):
                            #print('recording match in region',rgt_label)
                            pair_map[lft_idx,rgt_idx]=True

        # symmetrize the pairwise mask 
        pair_map = pair_map | pair_map.T
        # and do the actual swapping
        for i in range(label_subset.size):
            swapped=False # begin with no swap history for this region
            for j in range(i+1, label_subset.size): # previous entries are redundant.
                if pair_map[i,j]:
                    # print('relabeling',label_subset[i],'to',label_subset[j])
                    # j entries inherit swapability from i entries.
                    # only populate upper diagonal so that lower diagonal remains original
                    pair_map[j,j::] = pair_map[j,j::] | pair_map[i,j::] 
                    if not swapped: # if swapped already, there are no remaining i entries so we skip this.
                        i_msk = (label_array==label_subset[i])
                        # i label -> j label, which hasn't be processed
                        label_array[np.nonzero(i_msk)] = label_subset[j]
                        swapped=True
                        
        print('completed in',np.int32(time.time()-stime),'seconds')

        if in_place:
            return 0
        elif return_pairs:
            return label_array, pair_map
        else:
            return label_array

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
            # we need to do it again with the whole thing shifted in phi to allow for shorter separations on the periodic boundary
            roll_vol_seg = np.roll(self.result.vol_seg, np.int(self.inputs.nph/2), axis=0)
            roll_dist_i = ndi.distance_transform_edt( roll_vol_seg != self.result.labels[i] ).astype('float32')
            unroll_roll_dist_i = np.roll(roll_dist_i, -np.int(self.inputs.nph/2), axis=0)
            # and we combine both versions
            min_dist_i = np.minimum(dist_i, unroll_roll_dist_i)
            self.result.adj_msk[...,i] = ( min_dist_i <= ( np.float32(self.inputs.adj_thrsh) * self.result.hqv_width * self.source.crr / self.source.crr.mean() ) )
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


    def find_intersections(self, labels=None, group_name=None):
        
        ###################################################################################################
        # this method builds a branch structure showing the various groups with non-trivial intersections #
        ###################################################################################################

        import numpy as np

        if labels is None:
            labels=self.result.labels
            if group_name is None: group_name='all_regions'
        elif type(labels) is type(''):
            if labels=='all':
                labels=self.result.labels
                if group_name is None: group_name='all_regions'
            elif labels=='open':
                labels=self.result.open_labels
                if group_name is None: group_name='open_regions'
            elif labels=='clsd':
                labels=self.result.clsd_labels
                if group_name is None: group_name='clsd_regions'
            elif labels=='opos':
                labels=self.result.opos_labels
                if group_name is None: group_name='opos_regions'
            elif labels=='oneg':
                labels=self.result.oneg_labels
                if group_name is None: group_name='oneg_regions'
        elif type(labels)==type([]) or type(labels)==type(np.array(())):
            labels=np.array(labels)
            if group_name is None: group_name='custom_group'

        def group_type(labels=None):
            clsd = np.max([(label in self.result.clsd_labels)     for label in labels])
            opos = np.max([(label in self.result.opos_labels)     for label in labels])
            oneg = np.max([(label in self.result.oneg_labels)     for label in labels])
            iface_type=Foo()
            setattr(iface_type, 'clsd', clsd)
            setattr(iface_type, 'opos', opos)
            setattr(iface_type, 'oneg', oneg)
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
                    setattr(group_obj, 'status', None)
                    setattr(group_obj, 'labels', group_labels)
                    setattr(group_obj, 'volume', vol)
                    setattr(group_obj, 'top',    top)
                    setattr(group_obj, 'iface',  iface_type)
                    setattr(group_obj, 'n_regs', 2)
                    setattr(group_obj, 'children', [])
                    setattr(group_obj, 'parents', [])
                    group_list.append(group_obj)
                
        # now we explore depth with recursion

        print('Determining higher order overlap groups')
        n_regs=2
        new_groups=True # initialize number of groups to be explored.
        while new_groups:
            n_regs+=1
            print('Recursion level:',n_regs-2)
            new_groups = False
            label_len_list = [len(grp.labels) for grp in group_list]
            for i in range(len(group_list)):
                if group_list[i].status is None: # need to be explored
                    group_list[i].status = 'clear' # checked
                    subgroup_labels = group_list[i].labels
                    n_labels = len(subgroup_labels)
                    ss_same_len_list, = np.nonzero(np.array(label_len_list)==n_labels) # index of groups with same number of antries as current group
                    for j in labels[np.nonzero(labels > max(subgroup_labels))]: # next label to add to group
                        # first we make sure that j has nonempty overlaps with the elements of the group
                        nonempty_count = 0
                        parents=[i]
                        for k in range(n_labels):
                            test_labels = subgroup_labels.copy()
                            test_labels[k] = j # this is the branch group with the new index swapped in for one of the elements
                            test_labels = sorted(test_labels)
                            for ss_same_len in ss_same_len_list: # index of groups with same number of entries as current branch
                                if group_list[ss_same_len].labels == test_labels:
                                    nonempty_count+=1 # this swap works.
                                    parents.append(ss_same_len) # an equally valid parent entry
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
                                setattr(group_obj, 'status', None)
                                setattr(group_obj, 'labels', supergroup_labels)
                                setattr(group_obj, 'volume', vol)
                                setattr(group_obj, 'top',    top)
                                setattr(group_obj, 'iface',  iface_type)
                                setattr(group_obj, 'n_regs', n_regs)
                                setattr(group_obj, 'children', [])
                                setattr(group_obj, 'parents', sorted(parents))
                                group_list.append(group_obj)
                                # give each parent credit for the new group
                                for parent in parents:
                                    group_list[parent].children.append(len(group_list) - 1)
                                new_groups = True

        print('Overlap group search complete')
        # now we'll add the result to the list of groups.
        if self.result.intersections is None: self.result.intersections=Foo()
        if hasattr(self.result.intersections, group_name):
            print('Overwriting previous entry for group:',group_name)
        else:
            print('Populating new entry for group:',group_name)
        setattr(self.result.intersections, group_name, group_list)
        
        print('Success \n%%%%%%%%%%%%%%%%%%%%%%%%%%%%')

        return group_list
    
        ########################
        # end of method ########
        ########################





    def find_detached_HQVs(self, labels=None, group_name=None):
        
        ####################################################################################################
        # this method checks domain hqvs agains all hqv intersections to look for non-overlapping sections #
        ####################################################################################################

        import numpy as np
        from scipy import ndimage as ndi
        import skimage.measure as skim
        from skimage import morphology as mor

        if labels is None:
            labels=self.result.open_labels
            if group_name is None: group_name='open_regions'
        elif type(labels) is type(''):
            if labels=='all':
                labels=self.result.labels
                if group_name is None: group_name='all_regions'
            elif labels=='open':
                labels=self.result.open_labels
                if group_name is None: group_name='open_regions'
            elif labels=='clsd':
                labels=self.result.clsd_labels
                if group_name is None: group_name='clsd_regions'
            elif labels=='opos':
                labels=self.result.opos_labels
                if group_name is None: group_name='opos_regions'
            elif labels=='oneg':
                labels=[el for el in self.result.open_labels if el not in self.result.opos_labels]
                if group_name is None: group_name='oneg_regions'
        elif type(labels)==type([]) or type(labels)==type(np.array(())):
            labels=np.array(labels)
            if group_name is None: group_name='custom_group'
        
        group_list=[]

        print('Identifying non-interface HQVs')
        
        for reg in labels:

            # get the hqv volume that is not part of an interface
            reduced = list(self.result.labels[np.nonzero(self.result.labels != reg)])# all regions apart from the one specified...
            reg_hqv = self.get_reg_hqv(labels=[reg])
            ext_hqv = self.get_reg_hqv(labels=reduced, logic='Union')
            vol_seg = self.result.vol_seg

            # expand to allow growing through boundary
            if self.inputs.glbl_mdl:
                reg_hqv = self.global_expand(reg_hqv)
                ext_hqv = self.global_expand(ext_hqv)
                vol_seg = self.global_expand(vol_seg)
                
            # define auxilary masks
            reg_ifc = reg_hqv & ext_hqv
            reg_dtc = reg_hqv & ~reg_ifc
            #del(reg_hqv, ext_hqv)
            hqv_area = np.pi*(self.result.hqv_width)**2
            hqv_vol = (4*np.pi/3)*(self.result.hqv_width)**3

            # test volume for detached subvolumes
            if np.sum(reg_dtc) > hqv_vol:
                print('found detached hqv in excess of typical hqv volume for region',reg)
                dist_to_ifc = ndi.distance_transform_edt(~reg_ifc & (vol_seg==reg) )
                print('computed distances to interface')
                msk = (reg_dtc & (dist_to_ifc > self.result.hqv_width))
                msk = mor.closing(mor.opening(msk)) & ~reg_ifc
                if np.sum(msk)==0:
                    print('no significant detached features detected')
                else:
                    # discretely label objects
                    subvols = skim.label(msk)
                    # if global enforce periodicity and reduce to original grid
                    badregs = np.zeros(subvols.shape, dtype='bool')
                    sublabels = (np.unique(subvols[np.where(subvols!=0)]))

                    # remove small objects
                    
                    print(sublabels.size,'unique contiguous features found -- filtering')
                    for sublabel in sublabels:
                        msk = (subvols == sublabel)
                        msk_ind = np.where(subvols==sublabel)
                        if ((np.sum(msk) > hqv_vol) and (dist_to_ifc[msk_ind].mean() > self.result.hqv_width) and msk[...,-1].max()):
                            print('found significant detached segment in subregion',sublabel)
                        else:
                            subvols[msk_ind]=0
                            badregs[msk_ind]=True                       

                    old_sublabels = (np.unique(subvols[np.where(subvols!=0)]))
                    # test to see if there's anything to work with.
                    print(old_sublabels.size, 'unique contiguous feature(s) remain')
                    if old_sublabels.size == 0: # only if something left to do...
                        print('no significant features remain after filter')
                    else:
                        # surviving regions are grown back into mask
                        print('calculating distance to new features')
                        dist_to_reg = ndi.distance_transform_edt(subvols==0)
                        print('growing features within threshold distance')
                        reg_dtc_derivs = np.gradient(reg_dtc, axis=(0,1,2))
                        reg_dtc_overlap = (np.sqrt(reg_dtc_derivs[0]**2 + reg_dtc_derivs[1]**2 + reg_dtc_derivs[2]**2)!=0) & reg_ifc
                        subvols=mor.watershed(dist_to_reg, subvols, mask=((reg_dtc | reg_dtc_overlap) & ~badregs & (dist_to_reg < 2.0*self.result.hqv_width)))
                        del(reg_dtc_derivs, dist_to_reg)
                        # inclusion of the distance threshold limits growth along surface of ifc to keep structure compact.
                        # now we reduce if global
                        if self.inputs.glbl_mdl:
                            subvols = self.global_reduce(subvols)
                            vol_seg = self.global_reduce(vol_seg)
                            reg_hqv = self.global_reduce(reg_hqv)
                            ext_hqv = self.global_reduce(ext_hqv)
                            reg_ifc = self.global_reduce(reg_ifc)
                            reg_dtc = self.global_reduce(reg_dtc)
                            reg_dtc_overlap = self.global_reduce(reg_dtc_overlap)
                            self.associate_labels(subvols, axis=0, use_boundary=True)
                        # renumber to remove missing entries
                        old_sublabels = (np.unique(subvols[np.where(subvols!=0)])) 
                        swapped = np.zeros(subvols.shape, dtype='bool')
                        for i in range(old_sublabels.size):
                            ss = np.where((subvols==old_sublabels[i]) & ~swapped)
                            swapped[ss]=True
                            subvols[ss] = i+1
                        del(swapped, old_sublabels)
                        # now test overlap with other domains to see if fully detached or polar
                        fully_detached = []
                        pole_artefact = []
                        sublabels = np.unique(subvols[np.nonzero(subvols)])
                        for i in range(sublabels.size):
                            fully_detached.append(False)
                            pole_artefact.append(False)
                            submask = (subvols==sublabels[i])
                            if ( submask & reg_dtc_overlap ).max() == 0: fully_detached[i]=True
                            if ( submask[:,0,:].max() == 1 ) | ( submask[:,-1,:].max() == 1 ): pole_artefact[i]=True

                        dtc_obj = Foo()
                        setattr(dtc_obj, 'region', reg)
                        setattr(dtc_obj, 'subvol', subvols)
                        setattr(dtc_obj, 'sublabels', np.unique(subvols[np.nonzero(subvols)]))
                        setattr(dtc_obj, 'fully_detached', fully_detached)
                        setattr(dtc_obj, 'pole_artefact', pole_artefact)
                        group_list.append(dtc_obj)


        print('Detached HQV search complete')
        # now we'll add the result to the list of groups.
        if self.result.detached_HQVs is None: self.result.detached_HQVs=Foo()
        if hasattr(self.result.detached_HQVs, group_name):
            print('Overwriting previous entry for group: ',group_name)
        else:
            print('Populating new entry for group: ',group_name)
        setattr(self.result.detached_HQVs, group_name, group_list)
        
        print('Success \n%%%%%%%%%%%%%%%%%%%%%%%%%%%%')

        return group_list
    
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

        print('finding distances from nulls to discrete domains')

        if labels is None:
            print('region list not supplied -- selecting all')
            labels = self.result.labels
            labelname='all_regions'

        null_locs = self.source.null_locs
        N_null=null_locs.shape[1]

        if null_list is None:
            print('null list not supplied -- selecting all')
            null_list=np.arange(N_null)
            nullname='all_nulls'
        
        int_distance = np.zeros((np.size(null_list), np.size(labels))).astype('float32')
        hqv_distance = np.zeros((np.size(null_list), np.size(labels))).astype('float32')

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
                    hqv_distance[j,i] = hqv_box_dist
                else:
                    hqv_distance[j,i] = max(0.5, dist_mm[np.nonzero(hqv_msk)].min())
                int_msk = (self.result.vol_seg == labels[i])
                # test the typical int msk at the null box
                int_box_dist = np.mean(~int_msk[box_indices])
                if (int_box_dist < 0.5):
                    int_distance[j,i] = int_box_dist
                else:
                    int_distance[j,i] = max(0.5, dist_mm[np.nonzero(int_msk)].min())
                # with these conventions, the distance can go to zero, even if the null is not centered on a given pixel.
                #print('region: '+str(labels[i])+', null: '+str(n)+', hqv distance: '+str(hqv_distance[i,j])+', int distance:'+str(int_distance[i,j]))

        dist_object = Foo()
        setattr(dist_object, 'regions', np.int32(labels))
        setattr(dist_object, 'nulls', np.int32(null_list))
        setattr(dist_object, 'int_distance', np.float32(int_distance))
        setattr(dist_object, 'hqv_distance', np.float32(hqv_distance))

        if self.result.null_to_region_dist is None:
            self.result.null_to_region_dist = Foo()

        try:
            setattr(self.result.null_to_region_dist, labelname+'_'+nullname, dist_object)
        except AttributeError:
            print('cannot set attribute of null_to_region_dist')
            return

        return dist_object
        
        print('Success \n%%%%%%%%%%%%%%%%%%%%%%%%%%%%')

        


        ########################
        # end of method ########
        ########################


    def get_null_dtch_dist(self, groupname=None, null_list=None):

        ############################################################################################################
        # this method gets the distnaces from each null to each detached segment, in the form of a list of objects #
        ############################################################################################################

        import numpy as np

        print('finding distances from nulls to detached HQVs')
        
        if groupname is None:
            print('group not supplied -- selecting open')
            groupname = 'open_regions'

        if not hasattr(self.result.detached_HQVs, groupname):
            print(groupname,'not defined for list of detached HQVs')
            return

        null_locs = self.source.null_locs
        N_null=null_locs.shape[1]

        if null_list is None:
            print('null list not supplied -- selecting all')
            null_list=np.arange(N_null)
            nullname='all_nulls'
        
        hqv_distance = []

        for j in range(np.size(null_list)):
            hqv_distance.append([])
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
            domain_axis=[]
            sublabel_axis=[]
            for obj in getattr(self.result.detached_HQVs,groupname):
                # have to get the detached hqvs in the region, if any
                print('testing detached segments in region',obj.region)
                for sublabel in obj.sublabels:
                    print('checking distance to subregion',obj.region,':',sublabel)
                    domain_axis.append(obj.region)
                    sublabel_axis.append(sublabel)
                    # have to get the mask for the detached segment
                    hqv_msk = (obj.subvol==sublabel)
                    # test the typical hqv mask at the null box
                    hqv_box_dist = np.mean(~hqv_msk[box_indices])
                    if (hqv_box_dist < 0.5): # mostly inside domain.
                        hqv_distance[j].append(hqv_box_dist)
                    else:
                        hqv_distance[j].append(max(0.5, dist_mm[np.nonzero(hqv_msk)].min()))

        dist_object = Foo()
        setattr(dist_object, 'regions', np.int32(domain_axis))
        setattr(dist_object, 'subregions', np.int32(sublabel_axis))
        setattr(dist_object, 'nulls', np.int32(null_list))
        setattr(dist_object, 'hqv_distance', np.array(hqv_distance, dtype='float32'))

        if self.result.null_to_detached_dist is None:
            self.result.null_to_detached_dist = Foo()

        try:
            setattr(self.result.null_to_detached_dist, groupname+'_'+nullname, dist_object)
        except AttributeError:
            print('cannot set attribute of null_to_region_dist')
            return

        print('Success \n%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
        
        return dist_object

        


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
        self.cloan(model)
        print('Model keys: '+str([key for key in model.__dict__.keys()])+ ' read from '+str(fname))

        if self.result.adj_msk is None:
            pass
        else:
            if self.result.adj_msk_boolsize is not None and bitpack:
                print('Unpacking Adj mask due to bitwise storage on disk')
                self.result.adj_msk = numpy.array((numpy.unpackbits(self.result.adj_msk)[:self.result.adj_msk_boolsize]).reshape(self.result.adj_msk_shape), dtype='bool')

        return self
    
    def export_vtk(self, export_custom=None, export_model=None, fname=None, rr_rng=None, th_rng=None, ph_rng=None):

        from pyevtk.hl import gridToVTK
        import numpy as np

        if fname==None: fname=self.inputs.q_dir+'/3dvis'
        
        if export_model  is None: export_model = True # default to model export.

        # We'll transform the coordinates first
        crr=0.      + self.source.crr
        cth=np.pi/2 - self.source.cth*np.pi/180.
        cph=0.      + self.source.cph*np.pi/180.

        # First we need all of the coordinates in cartesian.
        cxx=crr*np.cos(cph)*np.sin(cth)
        cyy=crr*np.sin(cph)*np.sin(cth)
        czz=crr*np.cos(cth)

        # and the index ranges
        nrr = crr.shape[2]
        nth = crr.shape[1]
        nph = crr.shape[0]

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


        exportData = {}
        if export_model:
            bx=self.source.brr*np.sin(cth)*np.cos(cph) + self.source.bth*np.cos(cth)*np.cos(cph) + self.source.bph *(-np.sin(cph))
            by=self.source.brr*np.sin(cth)*np.sin(cph) + self.source.bth*np.cos(cth)*np.sin(cph) + self.source.bph *( np.cos(cph))
            bz=self.source.brr*np.cos(cth)             + self.source.bth*(-np.sin(cth))
            exportData['slog10q'] = self.source.slog10q[pl:pr,tl:tr,rl:rr].copy()
            exportData['bfield']  = (bx[pl:pr,tl:tr,rl:rr].copy(), by[pl:pr,tl:tr,rl:rr].copy(), bz[pl:pr,tl:tr,rl:rr].copy())
            exportData['vol_seg'] = self.result.vol_seg[pl:pr,tl:tr,rl:rr].copy()
            exportData['hqv_msk'] = self.result.hqv_msk[pl:pr,tl:tr,rl:rr].astype('uint8').copy()

        if export_custom is not None:
            try:
                if type(export_custom) is type(np.array(())):
                    print('exporting custom array')
                    if export_custom.dtype is np.dtype('bool'):
                        print('converting bool to u8')
                        exportData['custom'] = export_custom[pl:pr,tl:tr,rl:rr].astype('uint8').copy()
                    else:
                        exportData['custom'] = export_custom[pl:pr,tl:tr,rl:rr].copy()
                elif type(export_custom) is type({}):
                    print('exporting custom dicionary')
                    for key in export_custom.keys():
                        if type(export_custom[key]) is type(np.array(())):
                            print('dict el',key,'is array type')
                            if export_custom[key].dtype is np.dtype('bool'):
                                print('converting bool to u8')
                                exportData[key]=export_custom[key][pl:pr,tl:tr,rl:rr].astype('uint8').copy()
                            else:
                                exportData[key]=export_custom[key][pl:pr,tl:tr,rl:rr].copy()
            except:
                print('failed to index custom object')
                
        gridToVTK(fname, cxx[pl:pr,tl:tr,rl:rr].copy(), cyy[pl:pr,tl:tr,rl:rr].copy(), czz[pl:pr,tl:tr,rl:rr].copy(),
                  pointData = exportData)

        return
    



    
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
        opencolors = plt.cm.winter(openlin)
        clsdcolors = plt.cm.autumn(clsdlin)
        whitecolor = plt.cm.binary(0)
        blackcolor = plt.cm.binary(255)
        mycolors = np.vstack((np.flip(opencolors, axis=0),blackcolor,clsdcolors))
        cmap = mpl.colors.LinearSegmentedColormap.from_list('my_colormap', mycolors)

        return cmap

    def mskcmap(self, base_cmap):

        import numpy as np
        import pylab as plt
        import matplotlib as mpl

        from matplotlib.colors import ListedColormap
        if type(base_cmap) is str: base_cmap = plt.cm.get_cmap(base_cmap)
        else:
            print('argument must be a colarmap name string')
            return
        mskcmap = base_cmap(np.arange(base_cmap.N))
        mskcmap[:,-1]=np.linspace(0,1,base_cmap.N)
        mskcmap=ListedColormap(mskcmap)

        return mskcmap
        
    
    def visualize(self, window=None, figsize=None, cst_mask=None):

        import numpy as np
        import pylab as plt
        import matplotlib as mpl
        from matplotlib.widgets import Slider, RadioButtons, Button

        font = {'family':'serif', 'serif': ['computer modern roman']}
        plt.rc('text', usetex=True)
        plt.rc('font',**font)

        rr=None
        th=None
        ph=None
        
        P_null=self.source.null_locs.T
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
        dmask = np.zeros(dcube.shape, dtype='bool')
        ctable ='viridis'
        msk_ctable = self.mskcmap('spring_r')
        vrange = (-4,4)

        fig, axes = plt.subplots(num=window, nrows=2, ncols=2, gridspec_kw={'height_ratios': [(th_max-th_min),1.5*180/np.pi], 'width_ratios': [1.5*180/np.pi,(ph_max-ph_min)]})
        im1 = axes[0,0].imshow((dcube)[iph,:,:]  , vmin=vrange[0], vmax=vrange[1], cmap=ctable, extent=[rr_min,rr_max,th_min,th_max], origin='lower', aspect=(np.pi/180))
        im2 = axes[0,1].imshow((dcube)[:,:,irr].T, vmin=vrange[0], vmax=vrange[1], cmap=ctable, extent=[ph_min,ph_max,th_min,th_max], origin='lower', aspect=(1.))
        im3 = axes[1,1].imshow((dcube)[:,ith,:].T, vmin=vrange[0], vmax=vrange[1], cmap=ctable, extent=[ph_min,ph_max,rr_min,rr_max], origin='lower', aspect=(180./np.pi))
        im1m = axes[0,0].imshow((dmask)[iph,:,:]  , vmin=0, vmax=1, cmap=msk_ctable, extent=[rr_min,rr_max,th_min,th_max], origin='lower', aspect=(np.pi/180))
        im2m = axes[0,1].imshow((dmask)[:,:,irr].T, vmin=0, vmax=1, cmap=msk_ctable, extent=[ph_min,ph_max,th_min,th_max], origin='lower', aspect=(1.))
        im3m = axes[1,1].imshow((dmask)[:,ith,:].T, vmin=0, vmax=1, cmap=msk_ctable, extent=[ph_min,ph_max,rr_min,rr_max], origin='lower', aspect=(180./np.pi))
        if self.inputs.vis_title is not None:
            try: fig.suptitle(r'{0}'.format(self.inputs.vis_title), x=0.55, y=0.92, ha='center', fontsize=24)
            except: pass
        ch1a, = axes[0,0].plot([rr_min,rr_max], [th,th], '--', linewidth=1, color='black')
        ch1b, = axes[0,0].plot([lr,lr], [th_min,th_max], '--', linewidth=1, color='black')
        ch2a, = axes[0,1].plot([ph_min,ph_max], [th,th], '--', linewidth=1, color='black')
        ch2b, = axes[0,1].plot([ph,ph], [th_min,th_max], '--', linewidth=1, color='black')
        ch3a, = axes[1,1].plot([ph_min,ph_max], [lr,lr], '--', linewidth=1, color='black')
        ch3b, = axes[1,1].plot([ph,ph], [rr_min,rr_max], '--', linewidth=1, color='black')
        axes[0,0].set_ylabel(r'Latitude -- $\theta$ [deg]', fontsize=20)
        axes[0,0].set_xlabel(r'Radius -- $r/R_\odot [log]$', fontsize=20)
        axes[0,0].yaxis.set_ticks(np.linspace(-60, 60, 5 ))
        axes[0,0].xaxis.set_ticks(np.linspace(1,  2.5, 2 ))
        axes[0,1].yaxis.set_ticks(np.linspace(-60, 60, 5 ))
        axes[0,1].xaxis.set_ticks(np.linspace(0,  300, 6 ))
        axes[1,1].set_xlabel(r'Longitude -- $\phi$ [deg]', fontsize=20)
        axes[1,1].xaxis.set_ticks(np.linspace(0,  300, 6 ))
        axes[1,1].yaxis.set_ticks(np.linspace(1,  2.5, 2 ))
        axes[0,0].tick_params(axis='both', labelsize=16)
        axes[0,1].tick_params(axis='both', labelsize=16)
        axes[1,1].tick_params(axis='both', labelsize=16)
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
        cbar = fig.colorbar(im2, cax=cbar_ax, orientation = 'vertical')
        tickvals = np.linspace(vrange[0], vrange[1], 5)
        ticknames = ['{:2.1f}'.format(i) for i in tickvals]
        cbar.set_ticks(tickvals)
        cbar.ax.set_yticklabels(ticknames)
        cbar.ax.tick_params(labelsize=16)
        cbar.set_label(r'slog$_{10}Q_\perp$', rotation=90, fontsize=20)
        axcolor = 'lightgoldenrodyellow'
        axrr = plt.axes([0.125,   0.30, 0.12, 0.03], facecolor=axcolor)
        axth = plt.axes([0.125,   0.27, 0.12, 0.03], facecolor=axcolor)
        axph = plt.axes([0.125,   0.24, 0.12, 0.03], facecolor=axcolor)
        axbp = plt.axes([0.125,   0.20, 0.04, 0.03], facecolor=axcolor)
        axbr = plt.axes([0.165,   0.20, 0.04, 0.03], facecolor=axcolor)
        axbn = plt.axes([0.205,   0.20, 0.04, 0.03], facecolor=axcolor)
        axbt = plt.axes([0.249,   0.20, 0.04, 0.03], frameon=False)
        axbt.set_xticks([])
        axbt.set_yticks([])
        rax1 = plt.axes([0.125, 0.12, 0.04, 0.07], facecolor=axcolor)
        rax1.text(0.5, -0.3, r'Data',  ha='center')
        rax2 = plt.axes([0.165, 0.12, 0.04, 0.07], facecolor=axcolor)
        rax2.text(0.5, -0.3, r'Mask',  ha='center')
        rax3 = plt.axes([0.205, 0.12, 0.04, 0.07], facecolor=axcolor)
        rax3.text(0.5, -0.3, r'Options', ha='center')
        null_label = axbt.text(0,0,'')

        mask_key='hqv_msk'
        mask_opts='Disable'
        data_key='slog10q'
        inv_mask=False
        draw_params={'rr':rr, 'th':th, 'ph':ph, 'mask_key':mask_key, 'data_key':data_key, 'mask_opts':mask_opts, 'cst_mask':cst_mask}

        def redraw():

            nonlocal draw_params
            rr = draw_params['rr']
            th = draw_params['th']
            ph = draw_params['ph']
            data_key = draw_params['data_key']
            mask_key = draw_params['mask_key']
            mask_opts = draw_params['mask_opts']
            cst_mask = draw_params['cst_mask']
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
            draw_params={'rr':rr, 'th':th, 'ph':ph, 'mask_key':mask_key, 'data_key':data_key, 'mask_opts':mask_opts, 'cst_mask':cst_mask}

            if hasattr(self.source, data_key):
                dcube=getattr(self.source, data_key)
            if hasattr(self.result, data_key):
                dcube=getattr(self.result, data_key)
            if data_key=='brr':
                ctable='bwr_r'
                msk_ctable = self.mskcmap('binary')
                brms= np.around(np.sqrt((self.source.brr[...,irr]**2).mean()), decimals=2)
                vrange = (-brms, brms)
            elif data_key=='slog10q':
                ctable='viridis'
                msk_ctable = self.mskcmap('spring_r')
                vrange = (-4,4)
            elif data_key=='vol_seg':
                ctable = self.segcmap()
                msk_ctable = self.mskcmap('binary')
                vrange = (dcube.min(), dcube.max())
            else:
                print('unknown data key')
                return
            
            if mask_key is None:
                dmask=np.zeros(dcube.shape, dtype='bool')
            elif mask_key=='hqv_msk':
                dmask = self.result.hqv_msk
            elif mask_key=='seg_msk':
                dmask = ~self.result.seg_msk
            elif mask_key=='custom':
                if cst_mask is None:
                    dmask=np.zeros(dcube.shape, dtype='bool')
                else:
                    dmask = cst_mask
            else: pass
            
            if mask_opts=='Inverse':
                dmask=~dmask
            elif (mask_opts=='Disable') or (mask_opts is None):
                dmask=np.zeros(dcube.shape, dtype='bool')
            else: pass

            nonlocal im1, im2, im3, im1m, im2m, im3m, ch1a, ch1b, ch2a, ch2b, ch3a, ch3b, ar1, ar2, cbar, null_label

            im1.set_cmap(ctable)
            im2.set_cmap(ctable)
            im3.set_cmap(ctable)

            im1.set_data(dcube[iph,:,:])
            im2.set_data(dcube[:,:,irr].T)
            im3.set_data(dcube[:,ith,:].T)
            im1.set_clim(vmin=vrange[0], vmax=vrange[1])
            im2.set_clim(vmin=vrange[0], vmax=vrange[1])
            im3.set_clim(vmin=vrange[0], vmax=vrange[1])

            im1m.set_data(dmask[iph,:,:])
            im2m.set_data(dmask[:,:,irr].T)
            im3m.set_data(dmask[:,ith,:].T)

            im1m.set_cmap(msk_ctable)
            im2m.set_cmap(msk_ctable)
            im3m.set_cmap(msk_ctable)

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
                null_label=axbt.text(0,0.25,r'{0}'.format(str(c_null)))
            else:
                null_label=axbt.text(0,0.5,'')



            if data_key == 'vol_seg':
                tickvals = [dcube.min(), dcube.max()]
                ticknames = ['open', 'closed']
                cbarlabel = ''
            else:
                tickvals = np.linspace(vrange[0], vrange[1], 5)
                ticknames = ['{:2.1f}'.format(i) for i in tickvals]
                if data_key == 'slog10q':
                    cbarlabel='slog$_{10}Q_\perp$'
                elif data_key == 'brr'  :
                    cbarlabel='$B_r$ [Gauss]'
                    
            cbar.set_ticks(tickvals)
            cbar.ax.set_yticklabels(ticknames)
            cbar.set_label(r'{0}'.format(cbarlabel), rotation=90, fontsize=20)

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
            if label =='HQV':  draw_params['mask_key']='hqv_msk'
            if label =='$\delta \Omega_i$': draw_params['mask_key']='seg_msk'
            if label =='custom':   draw_params['mask_key']='custom'
            #print(mask_key)
            redraw()
            fig.canvas.draw_idle()

        def update_data_key(label):
            nonlocal draw_params
            if label=='$Q_\perp$':     draw_params['data_key']='slog10q'
            if label=='$\Omega_i$':      draw_params['data_key']='vol_seg'
            if label=='$B_r$': draw_params['data_key']='brr'
            #print(data_key)
            redraw()
            fig.canvas.draw_idle()

        def update_mask_opts(label):
            nonlocal draw_params
            if label=='Inverse': draw_params['mask_opts']='Inverse'
            if label=='Normal': draw_params['mask_opts']='Normal'
            if label=='Disable': draw_params['mask_opts']='Disable'
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

        srr = Slider(axrr, r'$r$',       1.0, 2.5,  valinit=rr, valfmt="%2.2f")
        sth = Slider(axth, r'$\theta$', -88., 88.,  valinit=th, valfmt="%2.1f")
        sph = Slider(axph, r'$\phi$'  ,  0.0, 360., valinit=ph, valfmt="%2.0f")

        data_selector_button = RadioButtons(rax1, (r'$\Omega_i$', r'$Q_\perp$', r'$B_r$'), active=1)    
        mask_selector_button = RadioButtons(rax2, (r'$\delta \Omega_i$', r'HQV', r'custom'), active=1)
        inv_maskersion_button = RadioButtons(rax3, (r'Inverse', r'Disable', r'Normal'), active=1)

        null_inc_button = Button(axbn, 'next', color='w', hovercolor='b')
        null_dec_button = Button(axbp, 'prev', color='w', hovercolor='b')
        reset_button = Button(axbr, 'null', color='r', hovercolor='b')
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
        inv_maskersion_button.on_clicked(update_mask_opts)

        #rrbox.on_submit(subrr)
        #thbox.on_submit(subth)
        #phbox.on_submit(subph)

        io_objects = (data_selector_button, mask_selector_button, inv_maskersion_button, null_inc_button, null_dec_button, reset_button)

        return fig, io_objects


###################################################
# a method the larger module for cloaning objects #

###################################################

def portattr(target_object, name, attribute):
    # this routine works like setattr but it detects model objects and instantiates from new
    namelist = ['Inputs', 'Source', 'Result', 'Foo']
    classname = attribute.__class__.__name__
    if classname in namelist: # attribute is a class object unto itself
        #print('Attribute', name, 'is of type', classname, 'and must be set dynamically')
        try:
            setattr(target_object, name, object.__new__(type(attribute))) # dynamically instantiate new instance
            getattr(target_object, name).__init__()
            for key in attribute.__dict__.keys():
                if key[0] != '_':
                    if key not in getattr(target_object, name).__dict__.keys() and classname is not 'Foo':
                        print('Setting unmatched attribute "',key,'" in object "',name,'"-- check version control.')
                    #print('Setting subattribute', key)
                    portattr(getattr(target_object, name), key, getattr(attribute, key))
        except:
            print('Cannot instantiate',name)
    else: # attribute is simply object
        #print('Attribute', name, 'is not an HQVseg object. Attempting to set statically')
        try:
            setattr(target_object, name, attribute)
            #print('Attribute', name, 'set statically.')
        except AttributeError:
            print('Attribute', name, 'cannot be set.')




            


    


