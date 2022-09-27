from dolfinx import fem
import ufl
#contains all weak formulations

def SUPG_weak_form(V1,u1,v1,c_func,K_vol,V2,u2,v2,K_vol_y,fy,c2_func,K2_vol,fx,K2_vol_x,is_wet=1,form='weak'):

    #adds in stability terms to weak forms
    #designed to be called inside CG_weak_form routine
    tau = fem.Function(V1)
    c1 = ufl.as_vector((c_func[0],c_func[1]))
    c2 = ufl.as_vector((c_func[2],c_func[3]))

    #fy4 + fy9
    K_vol.append(is_wet*tau*ufl.dot(c1,ufl.grad(u1))*ufl.dot(c1,ufl.grad(v1))*ufl.dx + is_wet*tau*u1*ufl.nabla_div(c1)*ufl.dot(c1,ufl.grad(v1))*ufl.dx)
    #fy5 + fy10
    K_vol.append(is_wet*tau*c_func[2]*v1*ufl.dot(c1,ufl.grad(u1))*ufl.dx + is_wet*tau*c_func[2]*u1*v1*ufl.nabla_div(c1)*ufl.dx)
    #fy6 + fy11
    K_vol.append(is_wet*tau*c_func[3]*v1*ufl.dot(c1,ufl.grad(u1))*ufl.dx + is_wet*tau*c_func[3]*u1*v1*ufl.nabla_div(c1)*ufl.dx)
    #fy7
    K_vol.append(is_wet*tau*c_func[2]*u1*ufl.dot(c1,ufl.grad(v1))*ufl.dx)
    #fy8
    K_vol.append(is_wet*tau*c_func[3]*u1*ufl.dot(c1,ufl.grad(v1))*ufl.dx)
    

    fy4 = fem.Function(V2)
    fy5 = fem.Function(V2)
    fy6 = fem.Function(V2)
    fy7 = fem.Function(V2)
    fy8 = fem.Function(V2)

    fy.append(fy4)
    fy.append(fy5)
    fy.append(fy6)
    fy.append(fy7)
    fy.append(fy8)

    K_vol_y += u2*v2*(fy4)*ufl.dx
    K_vol_y += u2*ufl.dot(ufl.grad(v2),ufl.as_vector((fy5,fy6)))*ufl.dx
    K_vol_y += v2*ufl.dot(ufl.grad(u2),ufl.as_vector((fy7,fy8)))*ufl.dx


    tau2 = fem.Function(V2)
    c2_2 = ufl.as_vector((c2_func[2],c2_func[3]))
    
    if form == 'weak':

        K2_vol = [tau2*ufl.dot(c2_2,ufl.grad(v2))*ufl.dot(c2_2,ufl.grad(u2))*ufl.dx + tau2*ufl.dot(c2_2,ufl.grad(v2))*u2*ufl.nabla_div(c2_2)*ufl.dx, tau2*c2_func[0]*u2*v2*ufl.nabla_div(c2_2)*ufl.dx,
                tau2*c2_func[1]*u2*v2*ufl.nabla_div(c2_2)*ufl.dx]

        K2_vol_x = is_wet*u1*v1*fx[0]*ufl.dx + is_wet*u1*ufl.dot(ufl.grad(v1),ufl.as_vector((fx[1],fx[2])))*ufl.dx
    if form == 'strong':
        
        K2_vol += [tau2*ufl.dot(c2_2,ufl.grad(v2))*ufl.dot(c2_2,ufl.grad(u2))*ufl.dx + tau2*ufl.dot(c2_2,ufl.grad(v2))*u2*ufl.nabla_div(c2_2)*ufl.dx, tau2*c2_func[0]*u2*v2*ufl.nabla_div(c2_2)*ufl.dx,
                tau2*c2_func[1]*u2*v2*ufl.nabla_div(c2_2)*ufl.dx]
        fx2 = fem.Function(V1)
        fx3 = fem.Function(V1)
        fx4 = fem.Function(V1)

        fx.append(fx2)
        fx.append(fx3)
        fx.append(fx4)
        K2_vol_x += is_wet*u1*v1*fx[1]*ufl.dx + is_wet*u1*ufl.dot(ufl.grad(v1),ufl.as_vector((fx[2],fx[3])))*ufl.dx
    #################################################################

    return tau,K_vol,fy,K_vol_y,tau2,c2_func,K2_vol,fx,K2_vol_x

def CG_weak_form(domain1,domain2,V1,V2,is_wet=1,SUPG='off'):
    #gives the weak form for the CG formulation
    #using 1 integration by parts
    n1 = ufl.FacetNormal(domain1)

    u1 = ufl.TrialFunction(V1)
    v1 = ufl.TestFunction(V1)


    cx_func = fem.Function(V1)
    cy_func = fem.Function(V1)
    csig_func = fem.Function(V1)
    cthet_func = fem.Function(V1)


    c_func = [cx_func,cy_func,csig_func,cthet_func]

    #create expressions and assemble linear forms which start in domain 1
    #need to separate boundary and volume integrals
    K_vol = [is_wet*cx_func*u1*v1.dx(0)*ufl.dx + is_wet*cy_func*u1*v1.dx(1)*ufl.dx,  is_wet*csig_func*u1*v1*ufl.dx, is_wet*cthet_func*u1*v1*ufl.dx]
    K_bound = [is_wet*ufl.dot(ufl.as_vector((cx_func,cy_func)),n1)*u1*v1*ufl.ds]  

    u2 = ufl.TrialFunction(V2)
    v2 = ufl.TestFunction(V2)
    n2 = ufl.FacetNormal(domain2)

    
    #now the resulting weak forms in domain 2
    fy1 = fem.Function(V2)
    fy2 = fem.Function(V2)
    fy3 = fem.Function(V2)

    fy = [fy1,fy2,fy3]

    #assemble the weak forms in domain 2 that depend on weak forms from K_vol
    K_vol_y = -u2*v2*fy[0]*ufl.dx
    K_vol_y += -ufl.inner(u2*ufl.as_vector((fy[1],fy[2])),ufl.grad(v2))*ufl.dx
    K_vol_y += u2*v2*ufl.dot(ufl.as_vector((fy[1],fy[2])),n2)*ufl.ds

    #assemble weak forms in domain 2 that depend on weak forms from K_bnd
    fy24 = fem.Function(V2) 
    K_bound_y = u2*v2*fy24*ufl.dx
    if  SUPG == 'off':
        #returns functions/vector of weak forms in first subdomain, corresponding functions and weak forms to be integrated in second,
        #then boundary boys and their corresponding second subdomain
        return c_func,K_vol,fy,K_vol_y,K_bound,[fy24],K_bound_y
    if  SUPG == 'on':
        #adds on stabilizing term and returns stuff
        
        c2_x = fem.Function(V2)
        c2_y = fem.Function(V2)
        c2_sig = fem.Function(V2)
        c2_thet = fem.Function(V2)
        c2 = [c2_x,c2_y,c2_sig,c2_thet]
        
        K2_vol = 0
        
        fx2 = fem.Function(V1)
        fx3 = fem.Function(V1)
        fx4 = fem.Function(V1)
        fx = [fx2,fx3,fx4]
        K2_vol_x = 0
        
        tau,K_vol,fy,K_vol_y,tau2,c2,K2_vol,fx,K2_vol_x = SUPG_weak_form(V1,u1,v1,c_func,K_vol,V2,u2,v2,K_vol_y,fy,c2,K2_vol,fx,K2_vol_x,is_wet=is_wet,form='weak')
        #tau,K_vol,fy,K_vol_y,tau2,c2,K2_vol,fx,K2_vol_x = SUPG_weak_form(V1,u1,v1,c_func,K_vol,V2,u2,v2,K_vol_y,fy,c2,K2_vol,fx,K2_vol_x,form='weak')   
        return c_func,K_vol,fy,K_vol_y,K_bound,[fy24],K_bound_y,tau,tau2,c2,K2_vol,fx,K2_vol_x

def SUPG_RHS(domain1,domain2,V1,V2,is_wet=1):
    #generates forms involving the RHS due to SUPG with implicit time stepping
    u1 = ufl.TrialFunction(V1)
    v1 = ufl.TestFunction(V1)
    tau = fem.Function(V1)
    u2 = ufl.TrialFunction(V2)
    v2 = ufl.TestFunction(V2)


    cx_func = fem.Function(V1)
    cy_func = fem.Function(V1)
    csig_func = fem.Function(V1)
    cthet_func = fem.Function(V1)


    c_func = [cx_func,cy_func,csig_func,cthet_func]
    
    L = [is_wet*tau*u1*ufl.dot(ufl.as_vector((cx_func,cy_func)),ufl.grad(v1))*ufl.dx, is_wet*tau*csig_func*u1*v1*ufl.dx, is_wet*tau*cthet_func*u1*v1*ufl.dx]

    g1 = fem.Function(V2)
    g2 = fem.Function(V2)
    g3 = fem.Function(V2)

    g = [g1,g2,g3]

    L_y = u2*v2*g1*ufl.dx + u2*ufl.dot(ufl.grad(v2),ufl.as_vector((g2,g3)))*ufl.dx

    return L,g,L_y,tau,c_func



def CG_strong_form(domain1,domain2,V1,V2,is_wet=1,SUPG='off'):
    #gives the weak form for the CG formulation
    #using 1 integration by parts
    n1 = ufl.FacetNormal(domain1)

    u1 = ufl.TrialFunction(V1)
    v1 = ufl.TestFunction(V1)


    cx_func = fem.Function(V1)
    cy_func = fem.Function(V1)
    csig_func = fem.Function(V1)
    cthet_func = fem.Function(V1)


    c_func = [cx_func,cy_func,csig_func,cthet_func]
   
   
    #create expressions and assemble linear forms which start in domain 1
    #in this case there are no boundary, just volume integrals
    K_vol = [is_wet*cx_func*u1.dx(0)*v1*ufl.dx + is_wet*cy_func*u1.dx(1)*v1*ufl.dx +  is_wet*ufl.nabla_div(ufl.as_vector((cx_func,cy_func)))*u1*v1*ufl.dx,  is_wet*csig_func*u1*v1*ufl.dx, is_wet*cthet_func*u1*v1*ufl.dx]


    #corresponding terms in the second subdomain
    u2 = ufl.TrialFunction(V2)
    v2 = ufl.TestFunction(V2)

    
    #now the resulting weak forms in domain 2
    fy1 = fem.Function(V2)
    fy2 = fem.Function(V2)
    fy3 = fem.Function(V2)

    fy = [fy1,fy2,fy3]

    #assemble the weak forms in domain 2 that depend on weak forms from K_vol
    K_vol_y = u2*v2*fy[0]*ufl.dx
    K_vol_y += ufl.inner(v2*ufl.as_vector((fy[1],fy[2])),ufl.grad(u2))*ufl.dx
    
    
    #create expressions that need to be integrated in subdomain 2 first
    c2_x = fem.Function(V2)
    c2_y = fem.Function(V2)
    c2_sig = fem.Function(V2)
    c2_thet = fem.Function(V2)
    c2 = [c2_x,c2_y,c2_sig,c2_thet]

    K2_vol = [ufl.nabla_div(ufl.as_vector((c2_sig,c2_thet)))*u2*v2*ufl.dx]

    fx1 = fem.Function(V1)
    
    fx = [fx1]

    K_vol_x = is_wet*u1*v1*fx[0]*ufl.dx
    
    if  SUPG == 'off':
        #returns functions/vector of weak forms in first subdomain, corresponding functions and weak forms to be integrated in second,
        #then boundary boys and their corresponding second subdomain
        return c_func,K_vol,fy,K_vol_y,c2,K2_vol,fx,K_vol_x
    if  SUPG == 'on':
        #adds on stabilizing term and returns stuff     
        tau,K_vol,fy,K_vol_y,tau2,c2,K2_vol,fx,K2_vol_x = SUPG_weak_form(V1,u1,v1,c_func,K_vol,V2,u2,v2,K_vol_y,fy,c2,K2_vol,fx,K_vol_x,is_wet=is_wet,form='strong')
        
        return c_func,K_vol,fy,K_vol_y,tau,tau2,c2,K2_vol,fx,K2_vol_x



