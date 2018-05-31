# SW model built over the structure of Ubelmann's QG model
# Version 1: Liu Yingjie, 2018
# To do list:
# Check boundary conditions


import numpy as np
from math import cos,sin,pi,isnan
from scipy.interpolate import griddata
import time
import numpy.matlib as matlib
import modgrid
import matplotlib.pylab as plt
import pdb
import lax_wendroff as lw


def sw(Hi=None, c=None, lon=None, lat=None, tint=None, dtout=None, dt=None,obsspace=None,Hm=None,rappel=None,snu=0.,depth=None):
    """ QG Shallow Water model

    Args:
        Hi (2D array): Initial SSH field.
        c (same size as Hi): Rossby first baroclinic phase speed
        lon (2D array): longitudes
        lat (2D array): latitudes
        tint (scalar): Time of propagator integration in seconds. Can be positive (forward integration) or negative (backward integration)
        dtout (scalar): Time period of outputs
        dt (scalar): Propagator time step

    Returns:
        SSH: 3D array with dimensions (timesteps, height, width), SSH forecast  SSH(timesteps,height,width)
    """
   # way=np.sign(tint)

  ##############
  # Setups
  ##############

    grd=modgrid.grid(Hi,c,snu,lon,lat)  ##very important parameter
    #plt.figure()
    #plt.pcolor(grd.mask)
    #plt.show()
    time_abs=0.
    index_time=0  
    if obsspace is not None:
        hg=np.empty((np.shape(obsspace)[0]))
        hg[:]=np.NAN
        iobs=np.where((way*obsspace[:,2]>=time_abs-dt/2) & (way*obsspace[:,2]<time_abs+dt/2))
        if np.size(iobs)>0:
            hg[iobs]=griddata((lon.ravel(), lat.ravel()), h.ravel(), (obsspace[iobs,0].squeeze(), obsspace[iobs,1].squeeze()))
    else:
        hg=None

    nindex_time=np.abs(tint)/dtout + 1
    SSH=np.empty((nindex_time,grd.ny,grd.nx))
    SSH[index_time,:,:]=Hi  
    
    nstep=int(abs(tint)/dt)
    stepout=int(dtout/dt)
   
  ############################
  # Active variable initializations   
  ############################
    u=np.zeros((grd.ny,grd.nx))
    v=np.zeros((grd.ny,grd.nx))
    H=np.zeros((grd.ny,grd.nx))
    cd=np.zeros((grd.ny,grd.nx))
    SS=np.empty((nindex_time))
    SS[index_time]=0. 
    g=grd.g
    F=grd.f0
    dx=grd.dx
    dy=grd.dy
    
    h=Hi+depth
    
    
    #if initially_geostrophic:
    """
    mat_contents = sio.loadmat('digital_elevation_map.mat')
    H = mat_contents['elevation'];
    H[[0, -1],:]=H[[1, -2],:];
    H[:,[0, -1]]=H[:,[1, -2]];
    """
    #v[:,:] = (g/(F[:,:]*dx)) * (h[:,:])
    
   # Centred spatial differences to compute geostrophic wind

   
    
    
    u[1:-1,:] = -(0.5*g/(F[1:-1,:]*dy[1:-1,:])) * (h[2:,:]-h[0:-2,:])
     
    v[:,1:-1] = (0.5*g/(F[:,1:-1]*dx[:,1:-1])) * (h[:,2:,]-h[:,0:-2])
    
    """
    #bd 0 gradient
    u[:,[0,-1]]=0
    u[[0,-1],1:-1]=u[[1,-2],1:-1]
    v[[0,-1],:]=0
    v[1:-1,[0,-1]]=v[1:-1,[1,-2]]
    h[[0,-1],:]=h[[1,-2],:]
    h[1:-1,[0,-1]]=h[1:-1,[1,-2]]
    """
    
    u[[0,-1],:]=u[[1,-2],:]
    u[:,[0,-1]]=u[:,[1,-2]]
    
    v[[0,-1],:]=v[[1,-2],:]
    v[:,[0,-1]]=v[:,[1,-2]]
    
    h[[0,-1],:]=h[[1,-2],:]
    h[1:-1,[0,-1]]=h[1:-1,[1,-2]]
    
    """
    #bd radiative
    u[[0 ,-1],:] = u[[1 ,-2],:];
    v[:,[0 ,-1]] = v[:,[1 ,-2]];
    #u[1:-1,[0 ,-1]] = u[1:-1,[1 ,-2]];
    #v[[0 ,-1],1:-1] = v[[1 ,-2],1:-1];
    """
    
    
    """
   #boundary conditions for initial wind
   # Zonal wind is periodic so set u(1) and u(end) as dummy points that
   # replicate u(end-1) and u(2), respectively    
    
    
    u[:,[0 ,-1]] = u[:,[1 ,-2]];
   # Meridional wind must be zero at the north and south edges of the
   # channel  
    v[[0, -1],:] = 0.;
    
    
   # Don't allow the initial wind speed to exceed 200 m/s anywhere
    max_wind = 200.
    u[np.where(u>max_wind)] = max_wind;
    u[np.where(u<-max_wind)] = -max_wind;
    v[np.where(v>max_wind)] = max_wind;
    v[np.where(v<-max_wind)] = -max_wind;
    u[np.where((np.isnan(u)))]=0
    v[np.where((np.isnan(v)))]=0
    """
    
    

  ############################
  # Time loop
  ############################
    
    for step in range(nstep): 
        #print step
        time_abs=(step+1)*dt
        if (np.mod(step+1,stepout)==0):
            index_time += 1

        ############################
        #Initialization of previous fields
        ############################

       
       

        ########################
        # Main routines
        ########################
          # Compute the accelerations H:bathymetry here H=0
       
        
        tu=snu*(dx[1:-1,0:-2]*u[1:-1,2:]-2*dx[1:-1,1:-1]*u[1:-1,1:-1]+dx[1:-1,2:]*u[1:-1,0:-2]) \
           /(dx[1:-1,1:-1]*dx[1:-1,2:]*dx[1:-1,0:-2]) \
        +snu*(dy[0:-2,1:-1]*u[2:,1:-1]-dy[1:-1,1:-1]*2*u[1:-1,1:-1]+dy[2:,1:-1]*u[0:-2,1:-1]) \
           /(dy[1:-1,1:-1]*dy[1:-1,0:-2]*dy[1:-1,2:])
      
        tv=snu*(dx[1:-1,0:-2]*v[1:-1,2:]-2*dx[1:-1,1:-1]*v[1:-1,1:-1]+dx[1:-1,2:]*v[1:-1,0:-2]) \
           /(dx[1:-1,1:-1]*dx[1:-1,2:]*dx[1:-1,0:-2]) \
        +snu*(dy[0:-2,1:-1]*v[2:,1:-1]-dy[1:-1,1:-1]*2*v[1:-1,1:-1]+dy[2:,1:-1]*v[0:-2,1:-1]) \
           /(dy[1:-1,1:-1]*dy[1:-1,0:-2]*dy[1:-1,2:])
        
        u_accel = F[1:-1,1:-1]*v[1:-1,1:-1] -(g/(2.*dx[1:-1,1:-1]))*(H[1:-1,2:]-H[1:-1,0:-2])+tu
             
        v_accel = -F[1:-1,1:-1]*u[1:-1,1:-1] - (g/(2.*dy[1:-1,1:-1]))*(H[2:,1:-1]-H[0:-2,1:-1])+tv
        
        
        #print 'test :',np.min(aaa), np.min(bbb), np.min(ccc), np.max(aaa), np.max(bbb), np.max(ccc)

          # Call the Lax-Wendroff scheme to move forward one timestep
        
       # (unew, vnew, h_new) = lw.lax_wendroff(dx, dy, dt, g, u, v, h, u_accel, v_accel);
        
        
        print 'step:',step,np.max(u),np.max(v),np.max(h)
        """
        plt.pcolormesh(h-Hi-depth)
        plt.colorbar(extend='both', fraction=0.042, pad=0.04)
        plt.show()
        """
        (unew, vnew, h_new) = lw.lax_wendroff(grd, dt, u, v, h, u_accel, v_accel,step,Hi,depth);
        
          
         # Update the wind and height fields, taking care to enforce 
         # boundary conditions 
        #print 'stat h:', np.min(h_new), np.max(h_new), np.argmin(h_new), np.argmax(h_new)
          
        """
           u = unew([end 1:end 1],[1 1:end end]);
           v = vnew([end 1:end 1],[1 1:end end]);
           v(:,[1 end]) = 0;
           h(:,2:end-1) = h_new([end 1:end 1],:);
        
        """
        """
        #periodic
        u[1:-1,1:-1] = unew[0:,0:];
        u[1:-1,[-1,0]]  = unew[:,[0,-1]]
        u[[0,-1],1:-1]  = unew[[0,-1],:]

        v[1:-1,1:-1] = vnew[0:,0:];
        v[0,[-1,0]]  = vnew[0,[0,-1]]
        v[[0,-1],1:-1]  = vnew[[0,-1],:]
   
        v[[0, -1],:] = 0.;

        h[1:-1,1:-1] = h_new[0:,0:];
        h[1:-1,[0,-1]]  = h_new[:,[-1,0]]
        """
        """
        #zero gradient
        u[:,[0,-1]]=0
        u[[0,-1],1:-1]=unew[[0,-1],:] 
        u[1:-1,1:-1] = unew[0:,0:]
        
      
        v[1:-1,[0,-1]]=vnew[:,[0,-1]] 
        v[[0,-1],:]=0
        v[1:-1,1:-1] = vnew[0:,0:]
        
         
        h[1:-1,[0,-1]]=h_new[:,[0,-1]] 
        h[[0,-1],1:-1]=h_new[[0,-1],:] 
        h[[0,-1,0,-1],[0,-1,-1,0]]=h_new[[0,-1,0,-1],[0,-1,-1,0]]
        h[1:-1,1:-1] = h_new[0:,0:]
        """
        
        u[1:-1,[0,-1]]=unew[:,[0,-1]] 
        u[[0,-1],1:-1]=unew[[0,-1],:] 
        u[[0,-1,0,-1],[0,-1,-1,0]]=unew[[0,-1,0,-1],[0,-1,-1,0]]
        u[1:-1,1:-1] = unew[0:,0:]
        
      
        v[1:-1,[0,-1]]=vnew[:,[0,-1]] 
        v[[0,-1],1:-1]=vnew[[0,-1],:]
        v[[0,-1,0,-1],[0,-1,-1,0]]=vnew[[0,-1,0,-1],[0,-1,-1,0]]
        v[1:-1,1:-1] = vnew[0:,0:]
        
         
        h[1:-1,[0,-1]]=h_new[:,[0,-1]] 
        h[[0,-1],1:-1]=h_new[[0,-1],:] 
        h[[0,-1,0,-1],[0,-1,-1,0]]=h_new[[0,-1,0,-1],[0,-1,-1,0]]
        h[1:-1,1:-1] = h_new[0:,0:]
       
        
        
        
        
        """
        # radiation
        
        
        u[:,[0]]= dt*((g*h[:,[0]])**0.5)*((u[:,[1]]-u[:,[0]])/dx[:,[0]])+u[:,[0]]   #dx 
        u[:,[-1]]=-dt*((g*h[:,[-1]])**0.5)*((u[:,[-1]]-u[:,[-2]])/dx[:,[-1]])+u[:,[-1]]  
        u[[0],1:-1]= dt*((g*h[[0],1:-1])**0.5)*((u[[1],1:-1]-u[[0],1:-1])/dy[[0],1:-1])+u[[0],1:-1]   #dy
        u[[-1],1:-1]=-dt*((g*h[[-1],1:-1])**0.5)*((u[[-1],1:-1]-u[[-2],1:-1])/dy[[-1],1:-1])+u[[-1],1:-1] 
        u[1:-1,1:-1] = unew[0:,0:]
        
        v[:,[0]]  = dt*((g*h[:,[0]])**0.5)*((v[:,[1]]-v[:,[0]])/dx[:,[0]])+v[:,[0]]   #dx
        v[:,[-1]]  =-dt*((g*h[:,[-1]])**0.5)*((v[:,[-1]]-v[:,[-2]])/dx[:,[-1]])+v[:,[-1]] 
        v[[0],1:-1]  = dt*((g*h[[0],1:-1])**0.5)*((v[[1],1:-1]-v[[0],1:-1])/dy[[0],1:-1])+v[[0],1:-1]   #dy
        v[[-1],1:-1]  =-dt*((g*h[[-1],1:-1])**0.5)*((v[[-1],1:-1]-v[[-2],1:-1])/dy[[-1],1:-1])+v[[-1],1:-1] 
        v[1:-1,1:-1] = vnew[0:,0:]
        
        h[:,[0]]  = dt*((g*h[:,[0]])**0.5)*((h[:,[1]]-h[:,[0]])/dx[:,[0]])+h[:,[0]]   #dx
        h[:,[-1]]  =-dt*((g*h[:,[-1]])**0.5)*((h[:,[-1]]-h[:,[-2]])/dx[:,[-1]])+h[:,[-1]] 
        h[[0],1:-1]  = dt*((g*h[[0],1:-1])**0.5)*((h[[1],1:-1]-h[[0],1:-1])/dy[[0],1:-1])+h[[0],1:-1]   #dy
        h[[-1],1:-1]  =-dt*((g*h[[-1],1:-1])**0.5)*((h[[-1],1:-1]-h[[-2],1:-1])/dy[[-1],1:-1])+h[[-1],1:-1]
        h[1:-1,1:-1] = h_new[0:,0:]
        """
       
        cd[:,:]=0.
        ############################
        #Saving outputs
        ############################
       
        if (np.mod(step+1,stepout)==0): 
            SSH[index_time,:,:]=h-depth
            
    return SSH



