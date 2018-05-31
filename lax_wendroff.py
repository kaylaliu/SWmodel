import numpy
import matplotlib.pylab as plt

def lax_wendroff(grd,dt, u, v, h, u_tendency, v_tendency,step,Hi,depth):
   
    g=grd.g
    dx=grd.dx
    dy=grd.dy
    #dx=dx[1:-1,1:-1]
    #dy=dy[1:-1,1:-1]
   # This function performs one timestep of the Lax-Wendroff scheme
   # applied to the shallow water equations

   # First work out mid-point values in time and space
 
    uh = u*h;
    vh = v*h;
    dx_mid=0.5*(dx[:,1:]+dx[:,0:-1])
    dy_mid=0.5*(dy[1:,:]+dy[0:-1,:])
    
    h_mid_yt = 0.5*(h[1:,:]+h[0:-1,:]) \
      -(0.5*dt/dy_mid)*(vh[1:,:]-vh[0:-1,:]);
    h_mid_xt = 0.5*(h[:,1:]+h[:,0:-1]) \
      -(0.5*dt/dx_mid)*(uh[:,1:]-uh[:,0:-1]);
    
   
    Ux = uh*u+0.5*g*h**2.;
    
    Uy = uh*v;
    uh_mid_yt = 0.5*(uh[1:,:]+uh[0:-1,:]) \
      -(0.5*dt/dy_mid)*(Uy[1:,:]-Uy[0:-1,:]);
    uh_mid_xt = 0.5*(uh[:,1:]+uh[:,0:-1]) \
      -(0.5*dt/dx_mid)*(Ux[:,1:]-Ux[:,0:-1]);

    Vx = Uy;
    Vy = vh*v+0.5*g*h**2.;
    vh_mid_yt = 0.5*(vh[1:,:]+vh[0:-1,:]) \
      -(0.5*dt/dy_mid)*(Vy[1:,:]-Vy[0:-1,:]);
    vh_mid_xt = 0.5*(vh[:,1:]+vh[:,0:-1]) \
      -(0.5*dt/dx_mid)*(Vx[:,1:]-Vx[:,0:-1]);
    
   # Now use the mid-point values to predict the values at the next
   # timestep
    h_new = h[1:-1,1:-1] \
      - (dt/dy[1:-1,1:-1])*(vh_mid_yt[1:,1:-1]-vh_mid_yt[0:-1,1:-1]) \
      - (dt/dx[1:-1,1:-1])*(uh_mid_xt[1:-1,1:]-uh_mid_xt[1:-1,0:-1]);
    h_1= h[1:-1,1] \
      - (dt/dy[1:-1,1])*(vh_mid_yt[1:,1]-vh_mid_yt[0:-1,1]) \
      - (dt/dx[1:-1,1])*(uh_mid_xt[1:-1,1]-uh_mid_xt[1:-1,0]);
    h_2= h[1:-1,2] \
      - (dt/dy[1:-1,2])*(vh_mid_yt[1:,2]-vh_mid_yt[0:-1,2]) \
      - (dt/dx[1:-1,2])*(uh_mid_xt[1:-1,2]-uh_mid_xt[1:-1,1]);
    h_3= h[1:-1,3] \
      - (dt/dy[1:-1,3])*(vh_mid_yt[1:,3]-vh_mid_yt[0:-1,3]) \
      - (dt/dx[1:-1,3])*(uh_mid_xt[1:-1,3]-uh_mid_xt[1:-1,2]);
    h_4= h[1:-1,4] \
      - (dt/dy[1:-1,4])*(vh_mid_yt[1:,4]-vh_mid_yt[0:-1,4]) \
      - (dt/dx[1:-1,4])*(uh_mid_xt[1:-1,4]-uh_mid_xt[1:-1,3]);
    h_5= h[1:-1,-2] \
      - (dt/dy[1:-1,-2])*(vh_mid_yt[1:,-2]-vh_mid_yt[0:-1,-2]) \
      - (dt/dx[1:-1,-2])*(uh_mid_xt[1:-1,-1]-uh_mid_xt[1:-1,-2]);
    """
    plt.pcolormesh(h_new-200-Hi[1:-1,1:-1])
    plt.colorbar(extend='both', fraction=0.042, pad=0.04)
    plt.show()
    """
    """
    h0=h_1-Hi[1:-1,1]-depth
    h00=h_2-depth-Hi[1:-1,2]
    h01=h_3-depth-Hi[1:-1,3]
    h11=h_4-depth-Hi[1:-1,4]
    h22=h_new[:,-1]-depth-Hi[1:-1,-2]
    plt.plot(h0)
    plt.show()
    plt.plot(h00)
    plt.show()
    plt.plot(h01)
    plt.show()
    plt.plot(h11)
    plt.show()
    plt.plot(h22)
    plt.show()
    """
    
    
   
    
    xx=- (dt/dy[1:-1,1:-1])*(vh_mid_yt[1:,1:-1]-vh_mid_yt[0:-1,1:-1]) \
       - (dt/dx[1:-1,1:-1])*(uh_mid_xt[1:-1,1:]-uh_mid_xt[1:-1,0:-1]);
    #print numpy.max(xx)
    Ux_mid_xt = uh_mid_xt*uh_mid_xt/h_mid_xt + 0.5*g*h_mid_xt**2.;
    Uy_mid_yt = uh_mid_yt*vh_mid_yt/h_mid_yt;
    uh_new = uh[1:-1,1:-1] \
      - (dt/dy[1:-1,1:-1])*(Uy_mid_yt[1:,1:-1]-Uy_mid_yt[0:-1,1:-1]) \
      - (dt/dx[1:-1,1:-1])*(Ux_mid_xt[1:-1,1:]-Ux_mid_xt[1:-1,0:-1]) \
      + dt*u_tendency*0.5*(h[1:-1,1:-1]+h_new);


    Vx_mid_xt = uh_mid_xt*vh_mid_xt/h_mid_xt;
    Vy_mid_yt = vh_mid_yt*vh_mid_yt/h_mid_yt + 0.5*g*h_mid_yt**2.;
    vh_new = vh[1:-1,1:-1] \
      - (dt/dy[1:-1,1:-1])*(Vy_mid_yt[1:,1:-1]-Vy_mid_yt[0:-1,1:-1]) \
      - (dt/dx[1:-1,1:-1])*(Vx_mid_xt[1:-1,1:]-Vx_mid_xt[1:-1,0:-1]) \
      + dt*v_tendency*0.5*(h[1:-1,1:-1]+h_new);
    u_new = uh_new/h_new;
    v_new = vh_new/h_new;
    
    #u_new[numpy.where((numpy.isnan(u_new)))]=0
    #v_new[numpy.where((numpy.isnan(v_new)))]=0
    #h_new[numpy.where((numpy.isnan(h_new)))]=0
    
    
    """
    if(step<=1000):
        for i in range(299):
            for j in range(173):
                if(abs(h_new[i,j]-Hi[i+1,j+1]-200)>=0.0001):
                    h_new[i,j]=Hi[i+1,j+1]+depth
    """       
    
    return (u_new, v_new, h_new)
