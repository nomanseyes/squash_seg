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
                 sbn_thrsh=None, ltq_thrsh=None, adj_thrsh=None, pad_ratio=None, bot_rad=None, \
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

    def __init__(self, crr=None, cth=None, cph=None, brr=None, bth=None, bph=None,  \
                 n_brr_bp=None, n_brr_bn=None, n_brr_tp=None, n_brr_tn=None, slog10q=None):

        # these are the basic model data: coordinates, bfield, slog10q

        self.crr=crr
        self.cth=cth
        self.cph=cph

        self.brr=brr
        self.bth=bth
        self.bph=bph

        self.n_brr_bp=n_brr_bp
        self.n_brr_bn=n_brr_bn
        self.n_brr_tp=n_brr_tp
        self.n_brr_tn=n_brr_tn
        
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

    def __init__(self, hqv_msk=None, PIL_ph=None, PIL_th=None, PIL_rr=None, GlnQp=None, reg_width=None, hqv_width=None, pad_msk=None, \
                 vol_seg=None, seg_msk=None, adj_msk=None, exterior_HQVs=None, interior_HQVs=None, \
                 null_to_region_dist=None, null_to_intHQV_dist=None, null_info=None, \
                 open_labels=None, clsd_labels=None, opos_labels=None, oneg_labels=None, labels=None):

        # and here are the new attributes, which are masks, intiger arrays, and lists of strings indicating groupings.

        # these come from the mask building

        self.hqv_msk            =hqv_msk
        self.PIL_ph             =PIL_ph
        self.PIL_th             =PIL_th
        self.PIL_rr             =PIL_rr
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
        self.exterior_HQVs         = exterior_HQVs
        self.interior_HQVs         = interior_HQVs
        self.null_info             = null_info
        self.null_to_region_dist   = null_to_region_dist
        self.null_to_intHQV_dist   = null_to_intHQV_dist


    ###########################
    # end of class definition #
    ###########################









class Model(object):

    # initialize the wrapper, with the various data objects as attributes
    # attributes are the initializtion arguments, the model data, and the post-processing results.
    # methods of this class will act on these objects.
    
    def __init__(self, model=None, inputs=None, source=None, result=None, auto_import=None, auto_segment=None, auto_group=None, auto_inspect=None, do_all=None, auto_save=None):

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
            auto_import  =True
            auto_segment =True
            auto_group   =True
            auto_inspect =True

        if auto_import:  self.do_import()
        if auto_segment: self.do_segment()
        if auto_group:   self.do_group()
        if auto_inspect: self.do_inspect()
        if auto_save:    self.save_data()
            
    def do_import(self):
        self.build_grid()
        self.import_squash_data()
        self.import_bfield_data()

    def do_segment(self):
        self.build_masks()
        self.segment_volume(visualize=True)

    def do_itemize(self):
        self.determine_adjacency()
        self.find_interior_HQVs()
        self.find_exterior_HQVs()

    def do_inspect(self):
        self.get_null_region_dist()
        self.get_null_intHQV_dist()
        self.associate_structures()
        self.categorize_structures()

    def do_all(self, auto_save=None):
        self.do_import()
        self.do_segment()
        self.do_itemize()
        self.do_inspect()
        if auto_save: self.save_data()

        
        

    #################################################
    # definitions of model methods ##################
    #################################################
    
    def cloan(self, donor, force=None):
        # a wrapper for porting all attributes
        for key in donor.__dict__.keys():
            if key in ['inputs', 'source', 'result']:
                portattr(self, key, getattr(donor, key), force=force)
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
            else:
                try: self.inputs.q_dir = os.path.abspath(self.inputs.q_dir)
                except TypeError: pass
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
        import skimage.measure as skim

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

            print('Checking Unipolar Regions')
            
            self.source.n_brr_bp = np.max(skim.label(b_rr_nat[..., 0] > 0))
            self.source.n_brr_bn = np.max(skim.label(b_rr_nat[..., 0] < 0))
            self.source.n_brr_tp = np.max(skim.label(b_rr_nat[...,-1] > 0))
            self.source.n_brr_tn = np.max(skim.label(b_rr_nat[...,-1] < 0))

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



    
    def build_masks(self, sbn_thrsh=None, ltq_thrsh=None, exclude_clip=None):

        ######################################################################
        # this method generates the mask that is used for the domain mapping #
        ######################################################################
        
        import numpy as np
        from skimage import morphology as mor

        # first get the PIL just for later use
        print('Building PIL masks')
        # bph -- derivs in rr and th
        dbph_dph, dbph_dth, dbph_drr = np.gradient(self.source.bph, np.pi*self.source.cph[:,0,0]/180, np.pi*self.source.cth[0,:,0]/180, self.source.crr[0,0,:], axis=(0,1,2))
        gdn_bph_ph = dbph_dph / np.cos(np.pi * self.source.cth / 180)
        gdn_bph_th = dbph_dth
        gdn_bph_rr = dbph_drr * self.source.crr
        self.result.PIL_ph = ( (gdn_bph_th**2 + gdn_bph_rr**2) > 10**4 * self.source.bph**2 )
        del(dbph_dph, dbph_dth, dbph_drr, gdn_bph_ph, gdn_bph_th, gdn_bph_rr)
        # bth -- derivs in rr and ph
        dbth_dph, dbth_dth, dbth_drr = np.gradient(self.source.bth, np.pi*self.source.cph[:,0,0]/180, np.pi*self.source.cth[0,:,0]/180, self.source.crr[0,0,:], axis=(0,1,2))
        gdn_bth_ph = dbth_dph / np.cos(np.pi * self.source.cth / 180)
        gdn_bth_th = dbth_dth
        gdn_bth_rr = dbth_drr * self.source.crr
        self.result.PIL_th = ( (gdn_bth_ph**2 + gdn_bth_rr**2) > 10**4 * self.source.bth**2 )
        del(dbth_dph, dbth_dth, dbth_drr, gdn_bth_ph, gdn_bth_th, gdn_bth_rr)
        # brr
        dbrr_dph, dbrr_dth, dbrr_drr = np.gradient(self.source.brr, np.pi*self.source.cph[:,0,0]/180, np.pi*self.source.cth[0,:,0]/180, self.source.crr[0,0,:], axis=(0,1,2))
        gdn_brr_ph = dbrr_dph / np.cos(np.pi * self.source.cth / 180)
        gdn_brr_th = dbrr_dth
        gdn_brr_rr = dbrr_drr * self.source.crr
        self.result.PIL_rr = ( (gdn_brr_ph**2 + gdn_brr_th**2) > 10**4 * self.source.brr**2 )
        del(dbrr_dph, dbrr_dth, dbrr_drr, gdn_brr_ph, gdn_brr_th, gdn_brr_rr)
        
        ### get the derivs.
        ### normalize derivs. want r*grad so deriv scales to radius
        ### in spherical, grad is d/dr, 1/r d/dth, 1/rsinth d/dph
        print('Calculating Qperp derivatives')
        ### we'll clip the datavalue before differentiating so that 10**value is within the range of float32
        ltqmax = np.log10(np.finfo('float32').max)-1
        clip_slog10q = np.clip(np.absolute(self.source.slog10q), -ltqmax, ltqmax)
        dslq_dph, dslq_dth, dslq_drr = np.gradient(np.log(10.) * clip_slog10q, np.pi*self.source.cph[:,0,0]/180, np.pi*self.source.cth[0,:,0]/180, self.source.crr[0,0,:], axis=(0,1,2))
        gdn_slq_ph = dslq_dph / np.cos(np.pi * self.source.cth / 180)
        del dslq_dph
        gdn_slq_th = dslq_dth
        del dslq_dth
        gdn_slq_rr = dslq_drr * self.source.crr
        del dslq_drr
        absQp = np.clip(10.**np.absolute(clip_slog10q), 2, 10**ltqmax)
    
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
                self.inputs.ltq_thrsh = 3.7 # can't get an RMS of Q b.c. the norm is infinite by definition... just choose something large (10**3.7 ~ 5000).
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





    def segment_volume(self, visualize=None):

        #########################################################
        # this method generates the domain map in the volume ####
        # metadata for the domain map are also generated ########
        #########################################################
        import pdb
        
        if self.result.hqv_msk is None: self.build_masks()
            
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


        vol_seg = np.zeros(hqv_msk.shape, dtype='int32')### initiate segmentation label array.
        # it's useful to have masks for open and closed flux. We consider undecided flux to be open.
        opn_msk = (slog10q < 0)
        if self.inputs.ss_eof: opn_msk[...,-1]=True
        cls_msk = (slog10q > 0)

        print('Calculating distance transforms')

        # it's also useful to have the distance from the interior of the hqv regions to their boundary, and the other way.
        # now we do our distance transform and masking
        hqv_dist = ndi.distance_transform_edt(~hqv_msk).astype('float32') # distance to nearest hqv
        reg_width = 4.*np.mean(hqv_dist[np.nonzero(~hqv_msk)]).astype('float32') # full width is 4 times average distance to boundary
        print('Vol width: ',reg_width)
        reg_dist = ndi.distance_transform_edt(hqv_msk).astype('float32') # distance to nearest low q region
        hqv_width = 4.*np.mean(reg_dist[np.nonzero(hqv_msk)]).astype('float32') # full width is 4 times average distance to boundary
        print('HQV width: ',hqv_width)
        del(reg_dist)

        print('Performing discrete flux labeling')

        if not self.inputs.pad_ratio: self.inputs.pad_ratio=0.2 # default is pad out to a typical quarter-width.

        # now we'll label the open flux domains, above min height.
        # we'll pad the hqv mask by a distance proportionate to the hqv_halfwidth, including a radial scaling to accomodate thicker hqvs at larger radii
        #pad_msk = ( hqv_dist > self.inputs.pad_ratio * hqv_width * (crr / crr.mean()) ) # empirical radial dependence...
        pad_msk = ( hqv_dist > self.inputs.pad_ratio * hqv_width ) # empirical radial dependence...
        open_label_mask = pad_msk & (slog10q < 0) & (crr >= self.inputs.bot_rad * self.inputs.solrad)
        clsd_label_mask = pad_msk & (slog10q > 0) & (crr >= self.inputs.bot_rad * self.inputs.solrad)
        vol_seg = np.zeros(pad_msk.shape, dtype='int32')
        vol_seg -= skim.label(open_label_mask).astype('int32') # all pixels not within or adjacent to a hqv
        vol_seg += skim.label(clsd_label_mask).astype('int32') # all pixels not within or adjacent to a hqv
        # and we'll define the label groups open and closed domains
        clsd_labels = np.unique(vol_seg[np.nonzero(vol_seg > 0)])
        open_labels = np.unique(vol_seg[np.nonzero(vol_seg < 0)])
        del(open_label_mask, clsd_label_mask)

        if visualize is None: visualize=False
        if visualize:
            import pylab as plt
            plt.figure(0)
            plt.imshow(hqv_msk[...,-1].T, origin='lower')
            plt.savefig(self.inputs.q_dir+'/hqv_msk.pdf')
            plt.figure(1)
            plt.imshow(pad_msk[...,-1].T, origin='lower')
            plt.savefig(self.inputs.q_dir+'/pad_msk.pdf')
            plt.figure(2)
            plt.imshow(slog10q[...,-1].T, origin='lower', vmax=4, vmin=-4)
            plt.savefig(self.inputs.q_dir+'/slog10q.pdf')
            plt.figure(3)
            plt.imshow(vol_seg[...,-1].T, origin='lower')
            plt.savefig(self.inputs.q_dir+'/vol_seg1.pdf')

        # Here we enforce periodicity
        # This MUST be done before removing domains and growing.
        # Otherwise small domains at the boundaries can be subsumed by larger domains.
        # This can then lead to erroneous cross-matching.        
        if self.inputs.glbl_mdl:
            print('Reconciling labels for periodicity')
            print('Clsd flux...')
            self.associate_labels(vol_seg, axis=0, label_subset=clsd_labels, use_volume=True, exp_loop=False)
            print('Open flux...')
            self.associate_labels(vol_seg, axis=0, label_subset=open_labels, use_volume=True, exp_loop=False)

        if visualize:
            plt.figure(4)
            plt.imshow(vol_seg[...,-1].T, origin='lower')
            plt.savefig(self.inputs.q_dir+'/vol_seg2.pdf')
        
        print('Removing domains with sub-minimum volume')
        crr_mean = crr.mean()
        hqv_vol = (4*np.pi/3)*(0.5*hqv_width)**3
        
        for reg in np.unique(vol_seg[np.nonzero(vol_seg)]):
            reg_ss = np.nonzero(vol_seg == reg)
            # threshold size for valid regions -- threshold increases quadratically with height to allow smaller closed domains
            # looks a bit odd. It's just sum over reg_ss.size, weighted by radial scaling, compared to hqv_vol. 
            if np.sum(crr_mean**2 / crr[reg_ss]**2) < hqv_vol: 
                vol_seg[reg_ss] = 0  # zero in mask, unchanged else.
        del(reg_ss, crr_mean, hqv_vol)

        if visualize:
            plt.figure(5)
            plt.imshow(vol_seg[...,-1].T, origin='lower')
            plt.savefig(self.inputs.q_dir+'/vol_seg3.pdf')

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

        if visualize:
            plt.figure(6)
            plt.imshow(vol_seg[...,-1].T, origin='lower')
            plt.savefig(self.inputs.q_dir+'/vol_seg4.pdf')
            

        print('Performing restricted watershed backfill into HQV mask')
        # Second, we grow regions using the same-type mask, which just expands open-open, close-close.
        stime = time.time()
        vol_seg += (mor.watershed(log10_SNQ, vol_seg * opn_msk, mask=opn_msk, watershed_line=False) * ((vol_seg==0) & opn_msk)).astype('int32')
        print('Open flux backfill completed in '+str(int(time.time()-stime))+' seconds')
        stime = time.time()
        vol_seg += (mor.watershed(log10_SNQ, vol_seg * cls_msk, mask=cls_msk, watershed_line=False) * ((vol_seg==0) & cls_msk)).astype('int32')
        print('Clsd flux backfill completed in '+str(int(time.time()-stime))+' seconds')

        if visualize:
            plt.figure(7)
            plt.imshow(vol_seg[...,-1].T, origin='lower')
            plt.savefig(self.inputs.q_dir+'/vol_seg5.pdf')

        print('Removing open domains without exterior footprint')
        # And we require that all regions are associated with a boundary
        # These loops are split up to allow for different definitions between open and closed domains
        opos_labels=[]
        oneg_labels=[]
        for ri in open_labels:
            ri_msk = (vol_seg[...,-1] == ri)
            # first we make sure there is an unmasked footprint
            if (ri_msk & ~hqv_msk[...,-1]).max():
                # open domains must have a net signed radial flux at the top boundary.
                # if the flux at the top is zero the region is invalid, and will be subsumed by an adjacent region.
                bi = np.sum(brr[...,-1][np.nonzero(ri_msk)])
                if bi > 0: opos_labels.append(ri)
                elif bi < 0: oneg_labels.append(ri)
                else: vol_seg[np.nonzero(vol_seg == ri)] = 0
            else: vol_seg[np.nonzero(vol_seg == ri)] = 0
        opos_labels = np.array(opos_labels)
        oneg_labels = np.array(oneg_labels)

        if visualize:
            plt.figure(8)
            plt.imshow(vol_seg[...,-1].T, origin='lower')
            plt.savefig(self.inputs.q_dir+'/vol_seg6.pdf')

        print('Performing transparent watershed backfill into HQV mask')
        # Third, we grow regions through opposite type, but only within a hqv, where type mixing is expected.
        stime = time.time()
        vol_seg += (mor.watershed(log10_SNQ, vol_seg * opn_msk, mask=hqv_msk, watershed_line=False, compactness=1) * ((vol_seg==0) & opn_msk)).astype('int32')
        print('Open flux backfill completed in '+str(int(time.time()-stime))+' seconds')
        stime = time.time()
        vol_seg += (mor.watershed(log10_SNQ, vol_seg * cls_msk, mask=hqv_msk, watershed_line=False, compactness=1) * ((vol_seg==0) & cls_msk)).astype('int32')
        print('Clsd flux backfill completed in '+str(int(time.time()-stime))+' seconds')

        if visualize:
            plt.figure(9)
            plt.imshow(vol_seg[...,-1].T, origin='lower')
            plt.savefig(self.inputs.q_dir+'/vol_seg7.pdf')

        print('Performing watershed backfill into residual domains')
        # Finally, we grow null regions with no preference, allowing open and closed to compete.
        stime = time.time()
        vol_seg += (mor.watershed(           1/(1 + hqv_dist), vol_seg,       watershed_line=False, compactness=1) * ((vol_seg==0))).astype('int32')
        vol_seg += (mor.watershed(           1/(1 + hqv_dist), vol_seg,       watershed_line=False, compactness=1) * ((vol_seg==0))).astype('int32')
        print('Final flux backfill completed in '+str(int(time.time()-stime))+' seconds')
        # There may still be unnasigned regions. But these will be outliers buried deep within opposite flux types.
        del(hqv_dist, opn_msk, cls_msk) # don't need these anymore

        if visualize:
            plt.figure(10)
            plt.imshow(vol_seg[...,-1].T, origin='lower')
            plt.savefig(self.inputs.q_dir+'/vol_seg8.pdf')
        
        print('Relabeling to remove obsolete domains') 
        # now let's relabel with integer labels removing gaps
        # first we need this list of labels that persist
        persistent_clsd_labels = np.unique(vol_seg[np.nonzero(vol_seg > 0)])
        persistent_open_labels = np.unique(vol_seg[np.nonzero(vol_seg < 0)])
        persistent_opos_labels = np.unique(list(set(opos_labels).intersection(set(persistent_open_labels))))
        persistent_oneg_labels = np.unique(list(set(oneg_labels).intersection(set(persistent_open_labels))))

        # now we generate new lists of same size with no gaps
        open_labels = - np.arange(1, persistent_open_labels.size + 1).astype('int32')
        opos_labels = - np.arange(1, persistent_opos_labels.size + 1).astype('int32')
        oneg_labels = - np.arange(1, persistent_oneg_labels.size + 1).astype('int32') + opos_labels.min() # offset by largest negative opos label
        clsd_labels = + np.arange(1, persistent_clsd_labels.size + 1).astype('int32')

        # and just quickly make these monotonic
        open_labels.sort()
        opos_labels.sort()
        oneg_labels.sort()
        clsd_labels.sort()

        # now we'll shuffle the ordering of the old ones to randomize the locations
        np.random.seed(persistent_open_labels.size) # repeatable random seed
        np.random.shuffle(persistent_open_labels) # random shuffle of domain order
        np.random.seed(persistent_opos_labels.size)
        np.random.shuffle(persistent_opos_labels)
        np.random.seed(persistent_oneg_labels.size)
        np.random.shuffle(persistent_oneg_labels)
        np.random.seed(persistent_clsd_labels.size)
        np.random.shuffle(persistent_clsd_labels) # same for clsd

        swapped = np.zeros(vol_seg.shape, dtype='bool') # boolean to track already swapped domains

        for i in np.arange(opos_labels.size):
            swap_msk = ((vol_seg == persistent_opos_labels[i]) & ~swapped)
            swapped  = swapped | swap_msk
            vol_seg[np.nonzero(swap_msk)] = opos_labels[i]
        
        for i in np.arange(oneg_labels.size):
            swap_msk = ((vol_seg == persistent_oneg_labels[i]) & ~swapped)
            swapped  = swapped | swap_msk
            vol_seg[np.nonzero(swap_msk)] = oneg_labels[i]
        
        for i in np.arange(clsd_labels.size):
            swap_msk = ((vol_seg == persistent_clsd_labels[i]) & ~swapped)
            swapped  = swapped | swap_msk
            vol_seg[np.nonzero(swap_msk)] = clsd_labels[i]
            
        del(swapped, swap_msk, persistent_opos_labels, persistent_oneg_labels, persistent_open_labels, persistent_clsd_labels)

        # get the whole label list anew
        labels = np.unique(vol_seg[np.nonzero(vol_seg)])
        # sanity checks:
        if set(labels) != set(open_labels).union(set(clsd_labels)):
            print('open and closed labels do not span the domain')
            pdb.set_trace()
        if set(open_labels) != set(opos_labels).union(set(oneg_labels)):
            print('opos and oneg labels do no span the open domain')
            pdb.set_trace()
        if set(opos_labels).intersection(set(oneg_labels)) != set([]):
            print('opos and oneg labels are not linearly independent')
            pdb.set_trace()
            
        if visualize:
            plt.figure(11)
            plt.imshow(vol_seg[...,-1].T, origin='lower')
            plt.savefig(self.inputs.q_dir+'/vol_seg9.pdf')

        # This completes the segmentation.
        print('Finished segmenting volume')

        
        # Now we just do a bit of book-keeping. 
        # get the pure interface boundaries from the domain map
        seg_gx, seg_gy, seg_gz = np.gradient(vol_seg, axis=(0,1,2))
        seg_msk = ((seg_gx**2 + seg_gy**2 + seg_gz**2) == 0)
        del(seg_gx, seg_gy, seg_gz)

        # restore the original shape of the domain.
        if self.inputs.glbl_mdl:
            print('reducing from double-size domain')
            vol_seg = self.global_reduce(vol_seg)
            pad_msk = self.global_reduce(pad_msk)
            hqv_msk = self.global_reduce(hqv_msk)
            slog10q = self.global_reduce(slog10q)
            crr     = self.global_reduce(crr)
            brr     = self.global_reduce(brr)
            GlnQp   = self.global_reduce(GlnQp)
            seg_msk = self.global_reduce(seg_msk)



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






    def associate_labels(self, input_array, axis=None, use_boundary=None, use_volume=None, lft_index_range=None, rgt_index_range=None, \
                         label_subset=None, mask=None, in_place=None, return_pairs=None, exp_loop=None):

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
                        
        print('label re-assignment completed in',np.int32(time.time()-stime),'seconds')

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
        self.result.adj_msk = np.zeros((nph, nth, nrr, nll), dtype='uint8')

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
            self.result.adj_msk[...,i] = ( min_dist_i <= ( np.float32(self.inputs.adj_thrsh) * self.result.hqv_width * (self.source.crr / self.source.crr.mean())**2 ) )
            if self.result.labels[i]==self.result.clsd_labels.max():
                print('Finished with Closed regions')
            if self.result.labels[i]==self.result.open_labels.max():
                print('Finished with Open regions')
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
            logic='Intersection'

        # note that the adjacency mask runs over non-zero entries to the label list.
        msk = (np.ones(self.result.hqv_msk.shape, dtype='bool') * (logic=='Intersection'))
        all_labels=self.result.labels
        for i in range(all_labels.size):
            if all_labels[i] in labels:
                if logic=='Union':
                    msk = msk | self.result.adj_msk[...,i] # unions
                elif logic=='Intersection':
                    msk = msk & self.result.adj_msk[...,i] # intersections
                else: print('hqv_type must be one of "Intersection" or "Union"')
                
        msk = msk & self.result.hqv_msk # project against hqv mask

        return msk


        

        ########################
        # end of method ########
        ########################





    def get_reg_bnd(self, labels=None, logic=None):
        
        ################################################################################################
        # this method extracts the boundary associated with a group of domains given a specified logic #
        ################################################################################################

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
            logic='Intersection'

        # note that the adjacency mask runs over non-zero entries to the label list.
        msk = (np.ones(self.result.hqv_msk.shape, dtype='bool') * (logic=='Intersection'))
        all_labels=self.result.labels
        for label in labels:
            reg = self.result.vol_seg == label
            bnd = get_mask_boundary(reg)
            if logic=='Union':
                msk = msk | bnd # unions
            elif logic=='Intersection':
                msk = msk & bnd # intersections
            else: print('hqv_type must be one of "Intersection" or "Union"')

        return msk
        

        ########################
        # end of method ########
        ########################


    def find_exterior_HQVs(self, labels=None, return_list=None, omit_clsd_pairs=None):
        
        ###################################################################################################
        # this method builds a branch structure showing the various groups with non-trivial intersections #
        ###################################################################################################

        import numpy as np
        from scipy import ndimage as ndi

        if labels is None:
            labels=self.result.labels

        if omit_clsd_pairs is None:
            omit_clsd_pairs = True
        
        def group_type(labels=None):
            clsd = np.max([(label in self.result.clsd_labels)     for label in labels])
            opos = np.max([(label in self.result.opos_labels)     for label in labels])
            oneg = np.max([(label in self.result.oneg_labels)     for label in labels])
            return clsd, opos, oneg

        group_list = []
        # labels need to be monotonic as this is assumed below.
        labels.sort()
        print('Inspecting ',np.size(labels),'regions in range ',labels[0],':',labels[-1])

        # first we append pairwise groupings
        print('Testing for overlapping regions...')
        if omit_clsd_pairs: print('                               ...skipping closed-only pairings')
        for l1 in labels:
            for l2 in labels[np.nonzero(labels>l1)]:
                # by default we ignore clsd-clsd pairings.
                clsd, opos, oneg = group_type(labels=[l1,l2])
                skip_clsd_pair = ( clsd and ( not (opos|oneg) ) and omit_clsd_pairs )
                if not skip_clsd_pair:
                    group_labels = [l1,l2]
                    print('                               ...testing pair:', group_labels, '          ',end='\r')
                    hqv = self.get_reg_hqv(labels=group_labels)
                    top = (np.sum(hqv[...,-1]) > 0)
                    bot = (np.sum(hqv[..., 0]) > 0)
                    vol = np.int32(np.sum(hqv))
                    pole = bool(hqv[:,0,:].max() | hqv[:,-1,:].max())
                    if vol > 0:
                        
                        # in clsd-field layers we don't really care about iface distinction
                        if clsd:
                            reg_iface=None
                        else:
                            # here we must distinguish hqv intersections from genuine domain interface layers
                            iface_mask = self.get_reg_bnd(labels = group_labels)
                            reg_iface = np.max(iface_mask)
                            
                        # now we get the volume subdivision
                        if reg_iface:
                            # now get distance measure to this interface
                            iface_dist = ndi.distance_transform_edt(~iface_mask)
                            # and make a threshold 'sausage' volume.
                            iface_ball = (iface_dist <= 2.*self.result.hqv_width)
                            del(iface_dist)
                            vol_frac = [np.float32(np.mean((self.result.vol_seg==label)[np.nonzero(iface_ball)])) for label in group_labels]
                        else:
                            vol_frac = [np.float32(np.mean((self.result.vol_seg==label)[np.nonzero(hqv       )])) for label in group_labels]

                        new_group_obj = Foo()
                        setattr(new_group_obj, 'status',     'new')
                        setattr(new_group_obj, 'labels',     group_labels)
                        setattr(new_group_obj, 'volume',     vol)
                        setattr(new_group_obj, 'vol_frac',   vol_frac)
                        setattr(new_group_obj, 'top',        top)
                        setattr(new_group_obj, 'bot',        bot)
                        setattr(new_group_obj, 'pole',       pole)
                        setattr(new_group_obj, 'opos',       opos)
                        setattr(new_group_obj, 'oneg',       oneg)
                        setattr(new_group_obj, 'clsd',       clsd)
                        setattr(new_group_obj, 'reg_iface',  reg_iface)
                        setattr(new_group_obj, 'n_regs',     2)
                        setattr(new_group_obj, 'children',   [])
                        setattr(new_group_obj, 'parents',    [])
                        group_list.append(new_group_obj)
        print('                               ...testing pair:', group_labels, '          ')
        print('                               ...finished')
        # now we explore depth with recursion

        print('Testing higher order groups...')
        n_regs=2
        new_groups=[group for group in group_list if (group.status is 'new')] # initialize number of groups to be explored.
        while new_groups:
            n_regs+=1
            for parent_group in new_groups:
                parent_group.status = 'clear'
                # it is assumed the labels are in monotonic order or they won't match the test templates.
                test_labels = labels[np.nonzero(labels > max(parent_group.labels))]
                for test_label in labels[np.nonzero(labels > max(parent_group.labels))]: # next label to add to group
                    group_labels = [parent_group.labels, [test_label]] # this needs flattened
                    group_labels = [el1 for el2 in group_labels for el1 in el2] # a bit opach, but it's just a double loop.
                    group_labels.sort()
                    clsd, opos, oneg = group_type(labels=group_labels)
                    print('                           ...group size:',n_regs,', testing group:', group_labels, '                              ',end='\r' )
                    # before we test the new group, make sure all of the parents have entries
                    all_parents_exist=True
                    parents = []
                    for el in group_labels:
                        coparent_labels = group_labels.copy()
                        coparent_labels.remove(el)
                        coparent_idx, = np.nonzero([ (group.labels == coparent_labels) for group in group_list])
                        if np.size(coparent_idx) is 0:
                            all_parents_exist=False
                        else:
                            parents.append(coparent_idx[0])
                    if all_parents_exist: # if all the parent elements are present, there could be a child intersection:
                        hqv = self.get_reg_hqv(labels=group_labels)
                        top = (np.sum(hqv[...,-1]) > 0)
                        bot = (np.sum(hqv[..., 0]) > 0)
                        vol = np.int32(np.sum(hqv))
                        pole = (hqv[:,0,:].max() | hqv[:,-1,:].max())
                        if vol > 0:
                            
                            # in clsd-field layers we don't really care about iface distinction
                            if clsd:
                                reg_iface=None
                            else:
                                # here we must distinguish hqv intersections from genuine domain interface layers
                                iface_mask = self.get_reg_bnd(labels = group_labels)
                                reg_iface = np.max(iface_mask)
                                
                            # now we get the volume subdivision
                            if reg_iface:
                                # now get distance measure to this interface
                                iface_dist = ndi.distance_transform_edt(~iface_mask)
                                # threshold distance makes 'sausage' volume
                                iface_ball = (iface_dist <= 2.*self.result.hqv_width)
                                del(iface_dist)
                                vol_frac = [np.float32(np.mean((self.result.vol_seg==label)[np.nonzero(iface_ball)])) for label in group_labels]
                            else:
                                vol_frac = [np.float32(np.mean((self.result.vol_seg==label)[np.nonzero(hqv       )])) for label in group_labels]
                            
                            new_group_obj = Foo()
                            setattr(new_group_obj, 'status',     'new')
                            setattr(new_group_obj, 'labels',     group_labels)
                            setattr(new_group_obj, 'volume',     vol)
                            setattr(new_group_obj, 'vol_frac',   vol_frac)
                            setattr(new_group_obj, 'top',        top)
                            setattr(new_group_obj, 'bot',        bot)
                            setattr(new_group_obj, 'pole',       pole)
                            setattr(new_group_obj, 'opos',       opos)
                            setattr(new_group_obj, 'oneg',       oneg)
                            setattr(new_group_obj, 'clsd',       clsd)
                            setattr(new_group_obj, 'reg_iface',  reg_iface)
                            setattr(new_group_obj, 'n_regs',     n_regs)
                            setattr(new_group_obj, 'children',   [])
                            setattr(new_group_obj, 'parents',    sorted(parents))
                            group_list.append(new_group_obj)
                            # give each parent credit for the new group
                            for parent in parents:
                                group_list[parent].children.append(np.size(group_list)-1)
                            
            new_groups=[group for group in group_list if (group.status is 'new')]
        print('                           ...group size:',n_regs,', testing group:', group_labels, '                              ')
        print('                           ...finished')

        print('Exterior HQV search complete')
        # now we'll add the result to the list of groups.
        if self.result.exterior_HQVs is not None: print('Overwriting previous entry')

        setattr(self.result, 'exterior_HQVs', group_list)
        
        print('Success \n%%%%%%%%%%%%%%%%%%%%%%%%%%%%')

        if return_list: return group_list
        
        ########################
        # end of method ########
        ########################



    

        
    def find_interior_HQVs(self, return_list=None):
        
        ####################################################################################################
        # this method checks domain hqvs agains all hqv intersections to look for non-overlapping sections #
        ####################################################################################################

        import numpy as np
        from scipy import ndimage as ndi
        import skimage.measure as skim
        from skimage import morphology as mor

        labels=self.result.open_labels
        
        group_list=[]

        print('Identifying non-interface HQVs')
        
        for reg in labels:

            # get the hqv volume that is not part of an interface
            reduced = list(self.result.labels[np.nonzero(self.result.labels != reg)])# all regions apart from the one specified...
            reg_msk = (self.result.vol_seg==reg)
            reg_hqv = self.get_reg_hqv(labels=[reg])
            ext_hqv = self.get_reg_hqv(labels=reduced, logic='Union')
            vol_seg = self.result.vol_seg

            # expand to allow growing through boundary
            if self.inputs.glbl_mdl:
                reg_msk = self.global_expand(reg_msk)
                reg_hqv = self.global_expand(reg_hqv)
                ext_hqv = self.global_expand(ext_hqv)
                vol_seg = self.global_expand(vol_seg)
                
            # define auxilary masks
            reg_ifc = reg_hqv & ext_hqv
            reg_dtc = reg_hqv & ~reg_ifc
            #del(reg_hqv, ext_hqv)
            hqv_area = np.pi*(self.result.hqv_width)**2
            hqv_vol = (4*np.pi/3)*(0.5*self.result.hqv_width)**3

            # test volume for detached subvolumes
            if np.sum(reg_dtc) > hqv_vol:
                print('found detached hqv in excess of typical hqv volume for region',reg)
                dist_to_ifc = ndi.distance_transform_edt(~reg_ifc & (vol_seg==reg) )
                #print('computed distances to interface')
                msk = (reg_dtc & (dist_to_ifc > 0.5*self.result.hqv_width))
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
                    #print(sublabels.size,'unique contiguous features found -- filtering')
                    for sublabel in sublabels:
                        msk = (subvols == sublabel)
                        msk_ind = np.where(subvols==sublabel)
                        if ((np.sum(msk) >= hqv_vol) and (dist_to_ifc[msk_ind].mean() >= 0.5*self.result.hqv_width) and msk[...,-1].max()):
                            print('found significant detached segment in subregion',sublabel)
                        else:
                            badregs[msk_ind]=True                       
                    #zero out regs that didn't pass the test
                    try: subvols[np.nonzero(badregs)] = 0
                    except IndexError: pass
                    
                    old_sublabels = (np.unique(subvols[np.where(subvols!=0)]))
                    # test to see if there's anything to work with.
                    #print(old_sublabels.size, 'unique contiguous feature(s) remain')
                    if old_sublabels.size == 0: # only if something left to do...
                        print('no significant features remain after filter')
                    else:
                        # surviving regions are grown back into mask
                        dist_to_sublabels = ndi.distance_transform_edt(subvols==0)
                        # only grow to pixels bounded by 1.5 x hqv_width from discrete labels (0.5 for origin thrshold and 1.0 for overlap)
                        subvols=mor.watershed(dist_to_sublabels, subvols, mask=(reg_msk & reg_hqv & (dist_to_sublabels <= 1.5*self.result.hqv_width)))
                        del(dist_to_sublabels)
                        # inclusion of the distance threshold limits growth along surface of ifc to keep structure compact.
                        # now we reduce if global
                        if self.inputs.glbl_mdl:
                            subvols = self.global_reduce(subvols)
                            vol_seg = self.global_reduce(vol_seg)
                            reg_msk = self.global_reduce(reg_msk)
                            reg_hqv = self.global_reduce(reg_hqv)
                            ext_hqv = self.global_reduce(ext_hqv)
                            reg_ifc = self.global_reduce(reg_ifc)
                            reg_dtc = self.global_reduce(reg_dtc)
                            #print('associating labels for periodicity')
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
                        sublabels = np.unique(subvols[np.nonzero(subvols)])
                        print(np.size(sublabels),'unique features remain after filtering')
                        for i in range(sublabels.size):
                            submask = (subvols==sublabels[i])
                            fully_detached = not bool( ( submask & reg_ifc ).max() ) 
                            polar_artefact = ( bool( submask[:,0,:].max() ) | bool( submask[:,-1,:].max() ) )
                            bot = ( np.sum(submask[..., 0]) > 0 )
                            hqv_obj = Foo()
                            setattr(hqv_obj, 'bot',            bot)
                            setattr(hqv_obj, 'label',          reg)
                            setattr(hqv_obj, 'sublabel',       sublabels[i])
                            setattr(hqv_obj, 'hqv_msk',        submask)
                            setattr(hqv_obj, 'fully_detached', fully_detached)
                            setattr(hqv_obj, 'polar_artefact', polar_artefact)
                            group_list.append(hqv_obj)


        print('Interior HQV search complete')
        # now we'll add the result to the list of groups.
        if self.result.interior_HQVs is not None:
            print('Overwriting previous entry')

        setattr(self.result, 'interior_HQVs', group_list)
        
        print('Success \n%%%%%%%%%%%%%%%%%%%%%%%%%%%%')

        if return_list: return group_list
    
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




    def get_null_region_dist(self):

        #################################################################################################
        # this method gets the distnaces from each null to each domain, in the form of a pair of arrays #
        #################################################################################################

        import numpy as np

        print('finding distances from nulls to discrete domains')

        try:
            labels = self.result.labels
        except AttributeError:
            'cannot retrieve label list'
            return -1

        null_locs = self.source.null_locs
        N_null=null_locs.shape[1]
        
        reg_distance = np.zeros((N_null, np.size(labels))).astype('float32')
        hqv_distance = np.zeros((N_null, np.size(labels))).astype('float32')

        for n in range(N_null):
            box = np.array(self.make_null_box(null_num=n, box_size=2)).T
            box_indices = (box[0], box[1], box[2])
            # we'll do measurements in sphericals but assuming orthonormality -- this is innacurate for large distances but we only care about small distances.
            # dividing by the metrics locally puts the distances into pixels, for easy comparison to the hqv_width, also in pixels.
            dist_rr = (self.source.crr - self.inputs.solrad*self.source.null_locs.T[n,2]) / self.source.metrr
            dist_th = self.source.crr*(np.pi / 180)*(self.source.cth - self.source.null_locs.T[n,1]) / self.source.metth 
            dist_ph = self.source.crr*np.cos(self.source.cth*np.pi/180)*(np.pi/180) * (self.source.cph - self.source.null_locs.T[n,0]) / self.source.metph
            dist_mm = np.sqrt(dist_rr**2 + dist_th**2 + dist_ph**2)
            del dist_rr, dist_th, dist_ph
            print('finding distances to null:',n,'            ',end='\r')
            for i in range(np.size(labels)):
                # have to get the mask for the region hqv
                hqv_msk = self.get_reg_hqv(labels=labels[i])
                # test the typical hqv mask at the null box
                hqv_box_dist = np.mean(~hqv_msk[box_indices])
                if (hqv_box_dist < 0.5): # mostly inside domain.
                    hqv_distance[n,i] = hqv_box_dist
                else:
                    hqv_distance[n,i] = max(0.5, dist_mm[np.nonzero(hqv_msk)].min())
                reg_msk = (self.result.vol_seg == labels[i])
                # test the typical int msk at the null box
                reg_box_dist = np.mean(~reg_msk[box_indices])
                if (reg_box_dist < 0.5):
                    reg_distance[n,i] = reg_box_dist
                else:
                    reg_distance[n,i] = max(0.5, dist_mm[np.nonzero(reg_msk)].min())
                # with these conventions, the distance can go to zero, even if the null is not centered on a given pixel.
                #print('region: '+str(labels[i])+', null: '+str(n)+', hqv distance: '+str(hqv_distance[i,j])+', int distance:'+str(int_distance[i,j]))
        print('finding distances to null:',n,'            ')

        dist_object = Foo()
        setattr(dist_object, 'labels', np.int32(labels))
        setattr(dist_object, 'nulls', np.int32(np.arange(N_null)))
        setattr(dist_object, 'reg_int_distance', np.float32(reg_distance))
        setattr(dist_object, 'reg_hqv_distance', np.float32(hqv_distance))

        try:
            setattr(self.result, 'null_to_region_dist', dist_object)
        except AttributeError:
            print('cannot set attribute of null_to_region_dist')
            return -1

        return dist_object
        
        print('Success \n%%%%%%%%%%%%%%%%%%%%%%%%%%%%')

        


        ########################
        # end of method ########
        ########################


    def get_null_intHQV_dist(self):

        ############################################################################################################
        # this method gets the distnaces from each null to each detached segment, in the form of a list of objects #
        ############################################################################################################

        import numpy as np

        print('finding distances from nulls to detached HQVs')

        null_locs = self.source.null_locs
        N_null=null_locs.shape[1]
        
        hqv_distance = []
        reg_distance = []

        for n in range(N_null):
            hqv_distance.append([])
            reg_distance.append([])
            box = np.array(self.make_null_box(null_num=n, box_size=2)).T
            box_indices = (box[0], box[1], box[2])
            # we'll do measurements in sphericals but assuming orthonormality -- this is innacurate for large distances but we only care about small distances.
            # dividing by the metrics locally puts the distances into pixels, for easy comparison to the hqv_width, also in pixels.
            dist_rr = (self.source.crr - self.inputs.solrad*self.source.null_locs.T[n,2]) / self.source.metrr
            dist_th = self.source.crr*(np.pi / 180)*(self.source.cth - self.source.null_locs.T[n,1]) / self.source.metth 
            dist_ph = self.source.crr*np.cos(self.source.cth*np.pi/180)*(np.pi/180) * (self.source.cph - self.source.null_locs.T[n,0]) / self.source.metph
            dist_mm = np.sqrt(dist_rr**2 + dist_th**2 + dist_ph**2)
            del dist_rr, dist_th, dist_ph
            print('finding distances to null:',n,'            ', end='\r')
            label_axis=[]
            sublabel_axis=[]
            for obj in self.result.interior_HQVs:
                # have to get the detached hqvs in the region, if any
                # print('checking distance to subregion',obj.region,':',sublabel)
                label_axis.append(obj.label)
                sublabel_axis.append(obj.sublabel)
                # test the hqv mask at the null box
                hqv_box_dist = np.mean(~obj.hqv_msk[box_indices])
                if (hqv_box_dist < 0.5): # mostly inside domain.
                    hqv_distance[n].append(hqv_box_dist)
                else:
                    hqv_distance[n].append(max(0.5, dist_mm[np.nonzero(obj.hqv_msk)].min()))
                # test the typical int msk at the null box
                reg_msk = (self.result.vol_seg == obj.label)
                reg_box_dist = np.mean(~reg_msk[box_indices])
                if (reg_box_dist < 0.5):
                    reg_distance[n].append(reg_box_dist)
                else:
                    reg_distance[n].append(max(0.5, dist_mm[np.nonzero(reg_msk)].min()))
                # print('distance to null:', hqv_distance[j][-1])
        print('finding distances to null:',n,'            ')
        dist_object = Foo()
        setattr(dist_object, 'labels', np.int32(label_axis))
        setattr(dist_object, 'sublabels', np.int32(sublabel_axis))
        setattr(dist_object, 'nulls', np.int32(np.arange(N_null)))
        setattr(dist_object, 'reg_hqv_distance', np.array(hqv_distance, dtype='float32'))
        setattr(dist_object, 'reg_int_distance', np.array(reg_distance, dtype='float32'))

        try:
            setattr(self.result, 'null_to_intHQV_dist', dist_object)
        except AttributeError:
            print('cannot set attribute of null_to_region_dist')
            return -1

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




        

    def associate_structures(self):

        # this routine is designed to connect the various disperate structures
        
        import numpy as np

        # first now we'll loop through exterior HQVs and check for nulls
        print('associating nulls to exterior HQVs')
        for hqv_obj in self.result.exterior_HQVs:
            associated_nulls = []
            for ni in self.result.null_to_region_dist.nulls:
                temp_obj = Foo()
                reg_hqv_dist_vec = []
                reg_int_dist_vec = []
                for label in hqv_obj.labels:
                    ri, = np.nonzero(label == self.result.null_to_region_dist.labels)
                    reg_hqv_dist_vec.append(self.result.null_to_region_dist.reg_hqv_distance[ni,ri][0])
                    reg_int_dist_vec.append(self.result.null_to_region_dist.reg_int_distance[ni,ri][0])
                reg_hqv_dist_rms = np.sqrt(np.mean(np.array(reg_hqv_dist_vec)**2))
                reg_int_dist_rms = np.sqrt(np.mean(np.array(reg_int_dist_vec)**2))
                setattr(temp_obj, 'null', ni)
                setattr(temp_obj, 'labels', hqv_obj.labels)
                setattr(temp_obj, 'reg_hqv_dist_vec', reg_hqv_dist_vec)
                setattr(temp_obj, 'reg_hqv_dist_rms', reg_hqv_dist_rms)
                setattr(temp_obj, 'reg_int_dist_vec', reg_int_dist_vec)
                setattr(temp_obj, 'reg_int_dist_rms', reg_int_dist_rms)
                associated_nulls.append(temp_obj)
            # now we reorder by hqv_dist_rms
            dlist = [el.reg_hqv_dist_rms for el in associated_nulls] 
            idxs = np.argsort(dlist)
            associated_nulls = [associated_nulls[idx] for idx in idxs]
            setattr(hqv_obj, 'associated_nulls', associated_nulls)

        # now we'll loop through interior HQVs and check for nulls
        print('associating nulls to interior HQVs')
        for hqv_obj in self.result.interior_HQVs:
            associated_nulls=[]
            # need to loop over the subregions for this specific region
            ri, = np.nonzero((hqv_obj.sublabel == self.result.null_to_intHQV_dist.sublabels) & (hqv_obj.label == self.result.null_to_intHQV_dist.labels))
            for ni in self.result.null_to_intHQV_dist.nulls:
                temp_obj = Foo()
                reg_hqv_dist = self.result.null_to_intHQV_dist.reg_hqv_distance[ni,ri][0]
                reg_int_dist = self.result.null_to_intHQV_dist.reg_int_distance[ni,ri][0]
                setattr(temp_obj, 'null', ni)
                setattr(temp_obj, 'label', hqv_obj.label)
                setattr(temp_obj, 'sublabel', hqv_obj.sublabel)
                setattr(temp_obj, 'reg_hqv_dist', reg_hqv_dist)
                setattr(temp_obj, 'reg_int_dist', reg_int_dist)
                associated_nulls.append(temp_obj)            
            # now we reorder by hqv_dist
            dlist = [el.reg_hqv_dist for el in associated_nulls] 
            idxs = np.argsort(dlist)
            associated_nulls = [associated_nulls[idx] for idx in idxs]
            setattr(hqv_obj, 'associated_nulls', associated_nulls)

        # finally, we'll tabulate a list of information organized by null number

        null_info = []

        print('tabulating associations by null index')
        
        for n in range(self.source.null_locs.shape[1]):
            null_info_obj = Foo()
            null_box = self.make_null_box(null_num=n, box_size=2)
            corner_labels = list(np.unique([self.result.vol_seg[loc] for loc in null_box]))
            associated_extHQVs = []
            associated_intHQVs = []
            extHQV_dist = []
            intHQV_dist = []
            #loop over ext HQVs and get null distances
            for hqv_obj in self.result.exterior_HQVs:
                for null_obj in hqv_obj.associated_nulls:
                    if (null_obj.null == n):
                        associated_extHQVs.append(hqv_obj)
                        extHQV_dist.append(null_obj.reg_hqv_dist_rms)
            #loop over int HQVs and get null distances
            for hqv_obj in self.result.interior_HQVs:
                for null_obj in hqv_obj.associated_nulls:
                    if (null_obj.null == n):
                        associated_intHQVs.append(hqv_obj)
                        intHQV_dist.append(null_obj.reg_hqv_dist)
            #sort by distance
            extHQV_idxs = np.argsort(extHQV_dist)
            intHQV_idxs = np.argsort(intHQV_dist)
            extHQV_dist = [extHQV_dist[idx] for idx in extHQV_idxs]
            intHQV_dist = [intHQV_dist[idx] for idx in intHQV_idxs]
            associated_extHQVs = [associated_extHQVs[idx] for idx in extHQV_idxs]
            associated_intHQVs = [associated_intHQVs[idx] for idx in intHQV_idxs]
            setattr(null_info_obj, 'null', n)
            setattr(null_info_obj, 'null_loc', self.source.null_locs[:,n])
            setattr(null_info_obj, 'corner_labels', corner_labels)
            setattr(null_info_obj, 'extHQVs', associated_extHQVs)
            setattr(null_info_obj, 'intHQVs', associated_intHQVs)
            setattr(null_info_obj, 'extHQV_dist', extHQV_dist)
            setattr(null_info_obj, 'intHQV_dist', intHQV_dist)
            null_info.append(null_info_obj)

        setattr(self.result, 'null_info', null_info)

        print('testing overlap of interior and exterior HQVs')
        
        # finally, we inspect the relationship between exterior and interior HQVs
        for int_hqv_obj in self.result.interior_HQVs: setattr(int_hqv_obj, 'associated_extHQVs', [])
        for ext_hqv_obj in self.result.exterior_HQVs: setattr(ext_hqv_obj, 'associated_intHQVs', [])

        n_int = np.size(self.result.interior_HQVs)
        n_ext = np.size(self.result.exterior_HQVs)
        for int_idx in range(n_int):
            int_hqv_obj = self.result.interior_HQVs[int_idx]
            for ext_idx in range(n_ext):
                ext_hqv_obj = self.result.exterior_HQVs[ext_idx]
                if (int_hqv_obj.label in ext_hqv_obj.labels) and (ext_hqv_obj.opos or ext_hqv_obj.oneg):
                    # check if all parents are  already associated with this int_hqv
                    # this shortest list of regions that need testing bc logicall the intersection of groups cannot be in if all groups are not already in
                    all_parents_in = True
                    for parent_idx in ext_hqv_obj.parents:
                        all_parents_in = all_parents_in * (self.result.exterior_HQVs[parent_idx] in int_hqv_obj.associated_extHQVs)
                    # if parents are in (or if there are no parents) we test the group
                    if ((ext_hqv_obj.n_regs == 2) or all_parents_in):
                        if (np.sum( int_hqv_obj.hqv_msk & self.get_reg_hqv(labels = ext_hqv_obj.labels) ) > 0):
                            print('... overlap found for ',int_hqv_obj.label,':',int_hqv_obj.sublabel,' against ',ext_hqv_obj.labels,'...')
                            int_hqv_obj.associated_extHQVs.append(self.result.exterior_HQVs[ext_idx])
                            ext_hqv_obj.associated_intHQVs.append(self.result.interior_HQVs[int_idx])
                    else:
                        pass


        print('success...')
        

        ########################
        # end of method ########
        ########################




    def categorize_structures(self, return_list=None):

        import numpy as np

        group_list = self.result.exterior_HQVs
        
        print('Inspecting exterior HQVs')
 
        #things that we want to know:
        #1 for n=2, is it a layer, or is it just a part of a vertex?
        #2 for n=2, if it is a layer: is it a branch, is it a simple layer, or is it w/in the HCS
        #3 for n>2, is it in the HCS or not? this does not depend on inherited information
        #3 for n>2, if not in HCS, is it a vertex? or is it just a merger? this also does not depend on inherited information.

        #strategy: vertices don't care whether their parents are branches/layers/etc. So we should work fron large n to small n

        # first the things that depend on parents, increasing order
        for group in group_list:
            group.conn_HCS=None # connects to HCS
            group.part_HCS=(group.top & group.opos & group.oneg) # part of HCS
            group.conn_OCB=None # connects to OCB
            group.part_OCB=(group.clsd & (group.opos | group.oneg)) # part of OCB
            group.conn_vtx=None # connects to vertex
            group.part_vtx=None # part of vertex -- in case of n=2, this implies vtx-only layer
            group.part_int=None # could be vertex, but less strict
            group.conn_int=None # could be vertex, but less strict
            group.merger  =None # like a vertex but too accute
            group.accute  =None # deals with opening angle in triplets
            group.simple  =None # boolean if simple layer
            group.branch  =None # boolean if branch layer
            group.cochild =None # index for cochild
            group.open_child=None # if has child with imprint at top boundary
            if (group.n_regs == 3):
                group.accute = bool( ( np.min(group.vol_frac) < 0.1) | ( np.min(group.vol_frac) < np.max(group.vol_frac)/6 ) )
                group.part_int = group.reg_iface and group.top and (not group.part_HCS)
                #print(group.labels,'vertex accusement pass:', group.part_vtx)
            if (group.n_regs >= 4):
                group.accute = np.max([group_list[parent].accute for parent in group.parents])
                parent_int_rat = np.mean([bool(group_list[parent].part_int) for parent in group.parents])
                group.part_int = (parent_int_rat >= 0.5) and group.top and (not group.part_HCS)
                #print(group.labels,'parent vertex accuteness pass:', group.part_vtx)
            # find cochildren -- i.e., the object at the bottom of the parent tree for a given group.

            # better name that cochild? what are we really talking about here!!!???
            # also, need one version w/o HCS and another w/ HCS
            # w/o deals with groupings w/in vertices
            # w/ deals with groupings w/in HCS intersections
            # want something like -- pair is part of an int which is w/in HCS, so pair is an artifact
            # same way we have for vetex... pair is part of int that is not/in HCS, so pair is an artifact.
            
            coparent_labels = [] # this will be populated later
            for child in group.children:
                if not (group_list[child].opos & group_list[child].oneg):
                    for coparent in group_list[child].parents:
                        if not group_list[coparent].clsd:
                            coparent_labels.append(group_list[coparent].labels)
            try:
                cochild = [el for el in group_list if ((el.labels == list(np.unique(coparent_labels))) and (el.n_regs > group.n_regs) and not el.clsd)][0]
                group.cochild = np.int32(np.nonzero([(el == cochild) for el in group_list]))[0][0]
            except IndexError:
                pass
                        

        # next, the things the depend on children, in decreasing order:
        for group in group_list[::-1]: # loop through backwards
            # basic inherited traits
            for child in group.children:
                if (not group_list[child].clsd):
                    group.open_child=True
                if group_list[child].part_int:
                    group.conn_int=True
                if group_list[child].part_HCS:
                    group.conn_HCS=True
                if group_list[child].part_OCB:
                    group.conn_OCB=True
                if group_list[child].part_vtx:
                    group.conn_vtx=True

            if (group.part_int and not group.conn_HCS):
                group.part_vtx = True 
                        
            # deeper inherited traits
            if group.cochild is not None:
                cochild = group_list[group.cochild]
                if group.top and cochild.top:
                    if not (group.part_HCS or group.conn_HCS or cochild.part_HCS or cochild.conn_HCS):
                        group.part_vtx = cochild.part_vtx
                    
            group.branch = (group.n_regs == 2) and group.top and group.reg_iface and group.conn_vtx and not (group.part_vtx or group.part_HCS)
            group.simple = (group.n_regs == 2) and group.top and group.reg_iface and group.conn_HCS and not (group.conn_vtx or group.part_vtx or group.part_HCS)
            group.vertex = group.top and group.part_vtx and not group.cochild
            group.hybrid = group.top and (group.simple or group.branch or group.vertex) and (np.size([el for el in group.associated_intHQVs if not el.polar_artefact]) > 0)
        # then, we filter out structures that are vertices whose parents are also vertices
        vtx_group = [el for el in group_list if el.vertex]
        for el1 in vtx_group:
            for parent in el1.parents:
                if group_list[parent] in vtx_group:
                    el1.vertex=False
                                
        setattr(self.result, 'exterior_HQVs', group_list)

        print('Inspecting interior HQVs')

        for obj in self.result.interior_HQVs:
            obj.conn_vertex = bool(np.size([el for el in obj.associated_extHQVs if el.vertex]))
            obj.conn_simple = bool(np.size([el for el in obj.associated_extHQVs if el.simple]))
            obj.conn_branch = bool(np.size([el for el in obj.associated_extHQVs if el.branch]))
            obj.conn_HCS    = bool(np.size([el for el in obj.associated_extHQVs if el.part_HCS]))
            obj.conn_OCB    = bool(np.size([el for el in obj.associated_extHQVs if el.part_OCB]))
            
        print('Success \n%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
        
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
        mskcmap[:,-1]=(np.linspace(0,1,base_cmap.N)>0.5).astype('float')
        mskcmap=ListedColormap(mskcmap)

        return mskcmap
        
    
    def visualize(self, window=None, figsize=None):

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

        dcube_rr = self.source.slog10q
        dcube_th = self.source.slog10q
        dcube_ph = self.source.slog10q
        dmask_rr = np.zeros(dcube_rr.shape, dtype='bool')
        dmask_th = dmask_rr
        dmask_ph = dmask_rr
        ctable ='viridis'
        msk_ctable = self.mskcmap('spring_r')
        vrange = (-4,4)

        fig, axes = plt.subplots(num=window, nrows=2, ncols=2, gridspec_kw={'height_ratios': [(th_max-th_min),1.5*180/np.pi], 'width_ratios': [1.5*180/np.pi,(ph_max-ph_min)]})
        im1 = axes[0,0].imshow((dcube_ph)[iph,:,:]  , vmin=vrange[0], vmax=vrange[1], cmap=ctable, extent=[rr_min,rr_max,th_min,th_max], origin='lower', aspect=(np.pi/180))
        im2 = axes[0,1].imshow((dcube_rr)[:,:,irr].T, vmin=vrange[0], vmax=vrange[1], cmap=ctable, extent=[ph_min,ph_max,th_min,th_max], origin='lower', aspect=(1.))
        im3 = axes[1,1].imshow((dcube_th)[:,ith,:].T, vmin=vrange[0], vmax=vrange[1], cmap=ctable, extent=[ph_min,ph_max,rr_min,rr_max], origin='lower', aspect=(180./np.pi))
        im1m = axes[0,0].imshow((dmask_ph)[iph,:,:]  , vmin=0, vmax=1, cmap=msk_ctable, extent=[rr_min,rr_max,th_min,th_max], origin='lower', aspect=(np.pi/180))
        im2m = axes[0,1].imshow((dmask_rr)[:,:,irr].T, vmin=0, vmax=1, cmap=msk_ctable, extent=[ph_min,ph_max,th_min,th_max], origin='lower', aspect=(1.))
        im3m = axes[1,1].imshow((dmask_th)[:,ith,:].T, vmin=0, vmax=1, cmap=msk_ctable, extent=[ph_min,ph_max,rr_min,rr_max], origin='lower', aspect=(180./np.pi))
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
        axbp = plt.axes([0.125,   0.21, 0.04, 0.03], facecolor=axcolor)
        axbr = plt.axes([0.165,   0.21, 0.04, 0.03], facecolor=axcolor)
        axbn = plt.axes([0.205,   0.21, 0.04, 0.03], facecolor=axcolor)
        axbt = plt.axes([0.249,   0.21, 0.04, 0.03], frameon=False)
        axbt.set_xticks([])
        axbt.set_yticks([])
        rax1 = plt.axes([0.125, 0.12, 0.04, 0.09], facecolor=axcolor)
        rax1.text(0.5, -0.3, r'Data',  ha='center')
        rax2 = plt.axes([0.165, 0.12, 0.04, 0.09], facecolor=axcolor)
        rax2.text(0.5, -0.3, r'Mask',  ha='center')
        rax3 = plt.axes([0.205, 0.12, 0.04, 0.09], facecolor=axcolor)
        rax3.text(0.5, -0.3, r'Tags', ha='center')
        null_label = axbt.text(0,0,'')

        mask_key=None
        tags_key='All'
        data_key='slog10q'
        draw_params={'rr':rr, 'th':th, 'ph':ph, 'mask_key':mask_key, 'data_key':data_key, 'tags_key':tags_key}

        imsk = np.sum([el.hqv_msk for el in self.result.interior_HQVs], axis=0).astype('bool')
        smsk = np.sum([self.get_reg_hqv(labels = el.labels) for el in self.result.exterior_HQVs if el.simple], axis=0).astype('bool')
        bmsk = np.sum([self.get_reg_hqv(labels = el.labels) for el in self.result.exterior_HQVs if el.branch], axis=0).astype('bool')
        vmsk = np.sum([self.get_reg_hqv(labels = el.labels) for el in self.result.exterior_HQVs if el.vertex], axis=0).astype('bool')
        hmsk = np.zeros(vmsk.shape, dtype='bool')
        for obj in [el for el in self.result.exterior_HQVs if el.hybrid]:
                    ext_msk = self.get_reg_hqv(labels = obj.labels)
                    int_msk = np.sum([el.hqv_msk for el in obj.associated_intHQVs], axis=0).astype('bool')
                    hmsk = hmsk | (ext_msk | int_msk)
        del(ext_msk, int_msk)

        def redraw():

            nonlocal draw_params
            rr = draw_params['rr']
            th = draw_params['th']
            ph = draw_params['ph']
            data_key = draw_params['data_key']
            mask_key = draw_params['mask_key']
            tags_key = draw_params['tags_key']
            
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
            draw_params={'rr':rr, 'th':th, 'ph':ph, 'mask_key':mask_key, 'data_key':data_key, 'tags_key':tags_key}

            # choose data
            if data_key=='bn':
                dcube_rr=getattr(self.source, 'brr')
                dcube_th=getattr(self.source, 'bth')
                dcube_ph=getattr(self.source, 'bph')
                ctable='bwr_r'
                msk_ctable = self.mskcmap('binary')
                brms= np.around(np.max((np.sqrt((self.source.brr[...,irr]**2).mean()), 0.1)), decimals=2)
                vrange = (-brms, brms)
            elif data_key=='slog10q':
                dcube_rr=getattr(self.source, 'slog10q')
                dcube_th=getattr(self.source, 'slog10q')
                dcube_ph=getattr(self.source, 'slog10q')
                ctable='viridis'
                msk_ctable = self.mskcmap('spring_r')
                vrange = (-4,4)
            elif data_key=='vol_seg':
                dcube_rr=getattr(self.result, 'vol_seg')
                dcube_th=getattr(self.result, 'vol_seg')
                dcube_ph=getattr(self.result, 'vol_seg')
                ctable = self.segcmap()
                msk_ctable = self.mskcmap('binary')
                vrange = (dcube_rr.min(), dcube_rr.max())
            elif data_key=='GlnQp':
                dcube_rr=getattr(self.result, 'GlnQp')
                dcube_th=getattr(self.result, 'GlnQp')
                dcube_ph=getattr(self.result, 'GlnQp')
                ctable='bone'
                msk_ctable = self.mskcmap('spring_r')
                vrange = (-50,50)
            else:
                print('unknown data key')
                return

            # choose mask
            if mask_key is None:
                dmask_ph=np.zeros(dcube_rr.shape, dtype='bool')
                dmask_th=dmask_ph
                dmask_rr=dmask_ph
            elif mask_key=='disable':
                dmask_ph=np.zeros(dcube_rr.shape, dtype='bool')
                dmask_th=dmask_ph
                dmask_rr=dmask_ph
            elif mask_key=='hqv_msk':
                dmask_ph=self.result.hqv_msk
                dmask_th=dmask_ph
                dmask_rr=dmask_ph
            elif mask_key=='seg_msk':
                dmask = ~self.result.seg_msk
                dmask_th=dmask_ph
                dmask_rr=dmask_ph
            elif mask_key=='PIL_msk':
                dmask_ph=self.result.PIL_ph
                dmask_th=self.result.PIL_th
                dmask_rr=self.result.PIL_rr

            # filter mask against tags
            if tags_key is 'all':
                pass
            elif tags_key=='intHQV':
                dmask_ph=dmask_ph & imsk
                dmask_th=dmask_th & imsk
                dmask_rr=dmask_rr & imsk
            elif tags_key=='simple':
                dmask_ph=dmask_ph & smsk
                dmask_th=dmask_th & smsk
                dmask_rr=dmask_rr & smsk
            elif tags_key=='branch':
                dmask_ph=dmask_ph & bmsk
                dmask_th=dmask_th & bmsk
                dmask_rr=dmask_rr & bmsk
            elif tags_key=='vertex':
                dmask_ph=dmask_ph & vmsk
                dmask_th=dmask_th & vmsk
                dmask_rr=dmask_rr & vmsk
            elif tags_key=='hybrid':
                dmask_ph=dmask_ph & hmsk
                dmask_th=dmask_th & hmsk
                dmask_rr=dmask_rr & hmsk

            nonlocal im1, im2, im3, im1m, im2m, im3m, ch1a, ch1b, ch2a, ch2b, ch3a, ch3b, ar1, ar2, cbar, null_label

            im1.set_cmap(ctable)
            im2.set_cmap(ctable)
            im3.set_cmap(ctable)

            im1.set_data(dcube_ph[iph,:,:])
            im2.set_data(dcube_rr[:,:,irr].T)
            im3.set_data(dcube_th[:,ith,:].T)
            im1.set_clim(vmin=vrange[0], vmax=vrange[1])
            im2.set_clim(vmin=vrange[0], vmax=vrange[1])
            im3.set_clim(vmin=vrange[0], vmax=vrange[1])

            im1m.set_data(dmask_ph[iph,:,:])
            im2m.set_data(dmask_rr[:,:,irr].T)
            im3m.set_data(dmask_th[:,ith,:].T)

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
                tickvals = [dcube_rr.min()/1.5, 0, dcube_rr.max()/1.5]
                ticknames = [r'Open', r'Mask', r'Closed']
                cbarlabel = ''
                cbar.set_ticks(tickvals)
                cbar.ax.set_yticklabels(ticknames, rotation=90,  verticalalignment='center', fontsize=20)
                cbar.set_label(r'{0}'.format(cbarlabel), rotation=90, fontsize=20)
            else:
                tickvals = np.linspace(vrange[0], vrange[1], 3)
                ticknames = ['{:2.1f}'.format(i) for i in tickvals]
                if data_key == 'slog10q':
                    cbarlabel='slog$_{10}Q_\perp$'
                elif data_key == 'bn'  :
                    cbarlabel='$B_n$ [Gauss]'
                elif data_key == 'GlnQp':
                    cbarlabel='$r (d/dr) \ln Q_\perp$'
                cbar.set_ticks(tickvals)
                cbar.ax.set_yticklabels(ticknames)
                cbar.set_label(r'{0}'.format(cbarlabel), rotation=90, fontsize=20)

            return 0

        def update_rr_coord(val, wait=None):
            nonlocal draw_params, null_vis, null_ini, rr_nodraw
            draw_params['rr'] = srr.val
            if (not rr_nodraw): # behavior when resetting a single coordinate allows immediate redraw and logic
                if (null_vis) & (not null_ini): null_vis=False
                null_ini=False
                if not wait: redraw()
                fig.canvas.draw_idle()
            else:
                rr_nodraw=False # alternative behavior to suppress redraw and logic for multiple coordinate updates
            return

        def update_th_coord(val, wait=None):
            nonlocal draw_params, null_vis, null_ini, th_nodraw
            draw_params['th'] = sth.val
            if (not th_nodraw):
                if (null_vis) & (not null_ini): null_vis=False
                null_ini=False
                if not wait: redraw()
                fig.canvas.draw_idle()
            else:
                th_nodraw=False
            return

        def update_ph_coord(val, wait=None):
            nonlocal draw_params, null_vis, null_ini, ph_nodraw
            draw_params['ph'] = sph.val
            if (not ph_nodraw):
                if (null_vis) & (not null_ini): null_vis=False
                null_ini=False
                if not wait: redraw()
                fig.canvas.draw_idle()
            else:
                ph_nodraw=False
            return

        def update_mask_key(label, wait=None):
            nonlocal draw_params
            if label =='HQV':               draw_params['mask_key']='hqv_msk'
            if label =='$\delta \Omega_i$': draw_params['mask_key']='seg_msk'
            if label =='PIL':               draw_params['mask_key']='PIL_msk'
            if label =='None':              draw_params['mask_key']='disable'
            if not wait: redraw()
            fig.canvas.draw_idle()

        def update_data_key(label, wait=None):
            nonlocal draw_params
            if label=='$\lg_{10} Q_\perp$':    draw_params['data_key']='slog10q'
            if label=='$\Omega_i$':            draw_params['data_key']='vol_seg'
            if label=='$B_n$':                 draw_params['data_key']='bn'
            if label=='G $\ln Q_\perp$':       draw_params['data_key']='GlnQp'
            if not wait: redraw()
            fig.canvas.draw_idle()

        def update_tags_key(label, wait=None):
            nonlocal draw_params
            if label=='all':    draw_params['tags_key']='all'
            if label=='intHQV': draw_params['tags_key']='intHQV'
            if label=='simple': draw_params['tags_key']='simple'
            if label=='branch': draw_params['tags_key']='branch'
            if label=='vertex': draw_params['tags_key']='vertex'
            if label=='hybrid': draw_params['tags_key']='hybrid'
            update_mask_key(label='HQV', wait=True)
            if not wait: redraw()
            fig.canvas.draw_idle()

        def increase_null_pos(value):
            nonlocal c_null, null_vis, null_ini, rr_nodraw, th_nodraw
            null_ini=True
            if (not null_vis): null_vis=True
            else: c_null = (c_null - 1)%N_null
            rr_nodraw=True
            th_nodraw=True
            srr.set_val(P_null[c_null,2])
            sth.set_val(P_null[c_null,1])
            sph.set_val(P_null[c_null,0])

        def decrease_null_pos(value):
            nonlocal c_null, null_vis, null_ini, rr_nodraw, th_nodraw
            null_ini=True
            if (not null_vis): null_vis=True
            else: c_null = (c_null + 1)%N_null
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

        data_selector_button  = RadioButtons(rax1, (r'$\lg_{10} Q_\perp$', r'G $\ln Q_\perp$', r'$\Omega_i$', r'$B_n$'), active=0)    
        mask_selector_button  = RadioButtons(rax2, (r'None', r'HQV', r'$\delta \Omega_i$', r'PIL'), active=0)
        tags_selector_button = RadioButtons(rax3, (r'all', r'intHQV', r'simple', r'branch', r'vertex', r'hybrid'), active=0)

        null_inc_button = Button(axbn, 'down', color='w', hovercolor='b')
        null_dec_button = Button(axbp, 'up', color='w', hovercolor='b')
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
        tags_selector_button.on_clicked(update_tags_key)

        #rrbox.on_submit(subrr)
        #thbox.on_submit(subth)
        #phbox.on_submit(subph)

        io_objects = (data_selector_button, mask_selector_button, tags_selector_button, null_inc_button, null_dec_button, reset_button)

        return fig, io_objects


###################################################
# a method the larger module for cloaning objects #

###################################################

def portattr(target_object, name, attribute, force=None):
    # this routine works like setattr but it detects model objects and instantiates from new
    if force is None: force=False
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
                        if force:
                            print('Setting unmatched attribute "',key,'" in object "',name,'"-- check version control.')
                            portattr(getattr(target_object, name), key, getattr(attribute, key))
                        else:
                            print('Refusing unmatched attribute "',key,'" in object "',name,'"-- check version control.')
                    else:
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





            
def get_mask_boundary(mask):
    
    # small routine to get the 2-pixel band that surrounds a discontinuity in a mask
        
    import numpy as np
    from scipy.ndimage import morphology as mor

    boundary = (~mask & mor.binary_dilation(mask, border_value=0) | (mask & ~mor.binary_erosion(mask, border_value=1)))

    return boundary




            


    


