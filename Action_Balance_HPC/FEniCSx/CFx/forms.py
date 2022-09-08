from dolfinx import fem
import ufl
#contains all weak formulations


def SUPG_weak_form(K_vol):
    #adds in stability terms to weak forms
    #designed to be called inside CG_weak_form routine
    
    return K_vol

def CG_weak_form(domain1,domain2,V1,V2,SUPG='off'):
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
    K_vol = [cx_func*u1*v1.dx(0)*ufl.dx + cy_func*u1*v1.dx(1)*ufl.dx,  csig_func*u1*v1*ufl.dx, cthet_func*u1*v1*ufl.dx]
    K_bound = [ufl.dot(ufl.as_vector((cx_func,cy_func)),n1)*u1*v1*ufl.ds]  

    u2 = ufl.TrialFunction(V2)
    v2 = ufl.TestFunction(V2)
    n2 = ufl.FacetNormal(domain2)

    
    #now the resulting weak forms in domain 2
    fy1 = fem.Function(V2)
    fy2 = fem.Function(V2)
    fy3 = fem.Function(V2)

    fy = [fy1,fy2,fy3]

    #assemble the weak forms in domain 2 that depend on weak forms from K_vol
    K2 = -u2*v2*fy[0]*ufl.dx
    K2 += -ufl.inner(u2*ufl.as_vector((fy[1],fy[2])),ufl.grad(v2))*ufl.dx
    K2 += u2*v2*ufl.dot(ufl.as_vector((fy[1],fy[2])),n2)*ufl.ds

    #assemble weak forms in domain 2 that depend on weak forms from K_bnd
    fy4 = fem.Function(V2) 
    K3 = u2*v2*fy4*ufl.dx
    if  SUPG == 'off':
        #returns functions/vector of weak forms in first subdomain, corresponding functions and weak forms to be integrated in second,
        #then boundary boys and their corresponding second subdomain
        return c_func,K_vol,fy,K2,K_bound,[fy4],K3
    if  SUPG == 'on':
        #adds on stabilizing term and returns stuff     
        K_vol = SUPG_weak_form(K_vol)
        return c_func,K_vol,fy,K2,K_bound,[fy4],K3

