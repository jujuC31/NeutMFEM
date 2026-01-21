"""
BENCHMARK IAEA 2D - Version adaptée pour NeutMFEM
Benchmark 2 groupes d'énergie avec réflecteur
"""

import time
import numpy as np
import argparse

import neutmfem._neutmfem as neutron_solver
from neutmfem._neutmfem import BCType, BoundaryAttribute, LinearSolverType

import seaborn as sns
import matplotlib.pyplot as plt

np.set_printoptions(threshold=np.inf, linewidth=np.inf)


class Iaea2D:
    """
    Benchmark IAEA 2D avec fonctionnalités avancées
    """
    
    def __init__(self, meshtype="2x2", domaine="entier", ncpu=1, sym="cyclique"):
        
        self.start = time.time()
        self.meshtype = meshtype
        self.domaine = domaine
        self.ncpu = ncpu
        self.sym = sym
        
        self.kref = 1.029585  # k-eff référence
        self.num_groups = 2
        self.verbose = 0
        self.order = 0
        
        self.init_meshing = False
        self.mysolv = None
        self.keff = None
        self.phi = None
        self.pvol = None

        # Maillage 19×19 assemblages
        self.maillage_motifs_coeur = np.array([
            ["  ", "  ", "  ", "  ", "  ", "  ", "  ", "  ", "  ", "  ", "  ", "  ", "  ", "  ", "  ", "  ", "  ", "  ", "  "],  
            ["  ", "  ", "  ", "  ", "  ", "  ", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "  ", "  ", "  ", "  ", "  ", "  "],  
            ["  ", "  ", "  ", "  ", "F4", "F4", "F4", "F1", "F1", "F1", "F1", "F1", "F4", "F4", "F4", "  ", "  ", "  ", "  "],  
            ["  ", "  ", "  ", "F4", "F4", "F1", "F1", "F1", "F2", "F2", "F2", "F1", "F1", "F1", "F4", "F4", "  ", "  ", "  "],  
            ["  ", "  ", "F4", "F4", "F1", "F1", "F2", "F2", "F2", "F2", "F2", "F2", "F2", "F1", "F1", "F4", "F4", "  ", "  "],  
            ["  ", "  ", "F4", "F1", "F1", "F3", "F2", "F2", "F2", "F3", "F2", "F2", "F2", "F3", "F1", "F1", "F4", "  ", "  "],  
            ["  ", "F4", "F4", "F1", "F2", "F2", "F2", "F2", "F2", "F2", "F2", "F2", "F2", "F2", "F2", "F1", "F4", "F4", "  "],
            ["  ", "F4", "F1", "F1", "F2", "F2", "F2", "F2", "F2", "F2", "F2", "F2", "F2", "F2", "F2", "F1", "F1", "F4", "  "],
            ["  ", "F4", "F1", "F2", "F2", "F2", "F2", "F2", "F2", "F2", "F2", "F2", "F2", "F2", "F2", "F2", "F1", "F4", "  "],
            ["  ", "F4", "F1", "F2", "F2", "F3", "F2", "F2", "F2", "F3", "F2", "F2", "F2", "F3", "F2", "F2", "F1", "F4", "  "],
            ["  ", "F4", "F1", "F2", "F2", "F2", "F2", "F2", "F2", "F2", "F2", "F2", "F2", "F2", "F2", "F2", "F1", "F4", "  "],
            ["  ", "F4", "F1", "F1", "F2", "F2", "F2", "F2", "F2", "F2", "F2", "F2", "F2", "F2", "F2", "F1", "F1", "F4", "  "],
            ["  ", "F4", "F4", "F1", "F2", "F2", "F2", "F2", "F2", "F2", "F2", "F2", "F2", "F2", "F2", "F1", "F4", "F4", "  "],
            ["  ", "  ", "F4", "F1", "F1", "F3", "F2", "F2", "F2", "F3", "F2", "F2", "F2", "F3", "F1", "F1", "F4", "  ", "  "],  
            ["  ", "  ", "F4", "F4", "F1", "F1", "F2", "F2", "F2", "F2", "F2", "F2", "F2", "F1", "F1", "F4", "F4", "  ", "  "],  
            ["  ", "  ", "  ", "F4", "F4", "F1", "F1", "F1", "F2", "F2", "F2", "F1", "F1", "F1", "F4", "F4", "  ", "  ", "  "],  
            ["  ", "  ", "  ", "  ", "F4", "F4", "F4", "F1", "F1", "F1", "F1", "F1", "F4", "F4", "F4", "  ", "  ", "  ", "  "],  
            ["  ", "  ", "  ", "  ", "  ", "  ", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "  ", "  ", "  ", "  ", "  ", "  "],  
            ["  ", "  ", "  ", "  ", "  ", "  ", "  ", "  ", "  ", "  ", "  ", "  ", "  ", "  ", "  ", "  ", "  ", "  ", "  "]
        ])

    def plot_geom(self):
        """Visualise la géométrie"""
        if not self.init_meshing:
            print("❌ Erreur: Le maillage doit être initialisé d'abord")
            return

        maillage_draw = []
        for row in self.maillage:
            maillage_draw.append([
                0 if cell == "  " else int(cell[1]) 
                for cell in row
            ])

        sns.heatmap(maillage_draw, cmap='jet', linewidths=0.5, linecolor="k")
        plt.title(f"Géométrie - {self.meshtype} - {self.domaine}")
        plt.show()

    def plot_materials(self):
        """Visualise les matériaux du cœur"""
        maillage_draw = []
        for row in self.maillage_motifs_coeur:
            maillage_draw.append([
                0 if cell == "  " else int(cell[1]) 
                for cell in row
            ])

        sns.heatmap(maillage_draw, cmap='jet', annot=True, 
                   linewidths=1, linecolor="k", fmt='d')
        plt.title("Distribution des matériaux IAEA 2D")
        plt.show()

    def mesh_initialisation(self, meshtype=None, domaine=None):
        """Initialise le maillage"""
        timeref = time.time()

        if meshtype:
            self.meshtype = meshtype
        if domaine:
            self.domaine = domaine
        
        if "x" in self.meshtype:
            self.nmeshes = int(self.meshtype.split("x")[0])
        else:
            self.nmeshes = len(self.meshtype)

        # Expansion
        self.maillage = np.array([
            [cell for cell in row for _ in range(self.nmeshes)] 
            for row in self.maillage_motifs_coeur 
            for _ in range(self.nmeshes)
        ])

        # Symétries
        L = len(self.maillage)
        L_half = L // 2
        
        domaine_map = {
            "quart_so": (slice(L_half, None), slice(None, L_half)),
            "quart_no": (slice(None, L_half), slice(None, L_half)),
            "quart_ne": (slice(None, L_half), slice(L_half, None)),
            "quart_se": (slice(L_half, None), slice(L_half, None)),
            "moitie_s": (slice(L_half, None), slice(None, None)),
            "moitie_o": (slice(None, None), slice(None, L_half)),
            "moitie_n": (slice(None, L_half), slice(None, None)),
            "moitie_e": (slice(None, None), slice(L_half, None)),
        }
        
        if self.domaine in domaine_map:
            y_slice, x_slice = domaine_map[self.domaine]
            self.maillage = self.maillage[y_slice, x_slice]

        # Coordonnées
        cell_size = 20.0 / self.nmeshes
        nx_cells = self.maillage.shape[1]
        ny_cells = self.maillage.shape[0]

        self.x_breaks = np.linspace(0.0, nx_cells * cell_size, nx_cells + 1)
        self.y_breaks = np.linspace(0.0, ny_cells * cell_size, ny_cells + 1)
        self.z_breaks = np.array([0.0])
        
        time1 = time.time()
        print(f"✅ Maillage initialisé: {ny_cells}×{nx_cells} cellules "
              f"({time1-timeref:.3f} s)")
        self.init_meshing = True

    def load_iaea2d_mat(self, include_upscattering=False):
        """Charge les matériaux avec upscattering optionnel"""
        
        # Matériau F1 - Combustible type 1
        self.F1 = {
            'D': [1.5, 0.4],
            'ABS': [0.010120, 0.080032],
            'NSF': [0.0, 0.135],
            'CHI': [1., 0.],
            'S12': 0.02
        }
        self.F1['SIGR'] = [self.F1['ABS'][0] + self.F1['S12'], self.F1['ABS'][1]]
        self.F1['S21'] = 0.0

        # Matériau F2 - Combustible type 2
        self.F2 = {
            'D': [1.5, 0.4],
            'ABS': [0.010120, 0.085032],
            'NSF': [0.0, 0.135],
            'CHI': [1., 0.],
            'S12': 0.02
        }
        self.F2['SIGR'] = [self.F2['ABS'][0] + self.F2['S12'], self.F2['ABS'][1]]
        self.F2['S21'] = 0.0

        # Matériau F3 - Combustible type 3 (avec barre de contrôle)
        self.F3 = {
            'D': [1.5, 0.4],
            'ABS': [0.010120, 0.130032],
            'NSF': [0.0, 0.135],
            'CHI': [1., 0.],
            'S12': 0.02
        }
        self.F3['SIGR'] = [self.F3['ABS'][0] + self.F3['S12'], self.F3['ABS'][1]]
        self.F3['S21'] = 0.0

        # Matériau F4 - Réflecteur
        self.F4 = {
            'D': [2.0, 0.3],
            'ABS': [0.000160, 0.010024],
            'NSF': [0.0, 0.0],
            'CHI': [0.0, 0.0],
            'S12': 0.04
        }
        self.F4['SIGR'] = [self.F4['ABS'][0] + self.F4['S12'], self.F4['ABS'][1]]
        self.F4['S21'] = 0.0

        # Matériau R0 - Réflecteur externe
        self.R0 = self.F4.copy()

        if include_upscattering:
            print("⚠️  Upscattering activé (non implémenté pour IAEA 2D)")

    def init_solver(self):
        """Initialise le solveur neutronique"""
        if not self.init_meshing:
            raise RuntimeError("Le maillage doit être initialisé avant le solveur")
        
        timeref = time.time()
        
        Ny, Nx = self.maillage.shape
        print(f"\n=== INITIALISATION SOLVEUR ({Ny}×{Nx}, {self.num_groups} groupes) ===")
        
        # Création du solveur
        self.mysolv = neutron_solver.NeutMFEM(
            self.order,
            self.num_groups,
            self.x_breaks,
            self.y_breaks,
            self.z_breaks
        )
        
        # Conditions aux limites par défaut
        self.mysolv.set_bc(BoundaryAttribute.LEFT_2D, BCType.DIRICHLET, 0.0)
        self.mysolv.set_bc(BoundaryAttribute.RIGHT_2D, BCType.DIRICHLET, 0.0)
        self.mysolv.set_bc(BoundaryAttribute.TOP_2D, BCType.DIRICHLET, 0.0)
        self.mysolv.set_bc(BoundaryAttribute.BOTTOM_2D, BCType.DIRICHLET, 0.0)

        # Application des symétries
        if self.domaine == "quart_so":
            self.mysolv.apply_central_symmetry(1, 1)
            self.mysolv.set_bc(BoundaryAttribute.TOP_2D, BCType.MIRROR)
            self.mysolv.set_bc(BoundaryAttribute.RIGHT_2D, BCType.MIRROR)
            self.mysolv.set_bc(BoundaryAttribute.LEFT_2D, BCType.DIRICHLET, 0.0)
            self.mysolv.set_bc(BoundaryAttribute.BOTTOM_2D, BCType.DIRICHLET, 0.0)
            print("  Domaine QUART_SO : Symétrie centrale activée")
        
        elif self.domaine == "moitie_s":
            self.mysolv.apply_central_symmetry(1, 0)
            self.mysolv.set_bc(BoundaryAttribute.TOP_2D, BCType.MIRROR)
            self.mysolv.set_bc(BoundaryAttribute.LEFT_2D, BCType.DIRICHLET, 0.0)
            self.mysolv.set_bc(BoundaryAttribute.RIGHT_2D, BCType.DIRICHLET, 0.0)
            self.mysolv.set_bc(BoundaryAttribute.BOTTOM_2D, BCType.DIRICHLET, 0.0)
            print("  Domaine MOITIE_S : Symétrie centrale activée")
        
        elif self.domaine == "moitie_o":
            self.mysolv.apply_central_symmetry(1, 0)
            self.mysolv.set_bc(BoundaryAttribute.RIGHT_2D, BCType.MIRROR)
            self.mysolv.set_bc(BoundaryAttribute.LEFT_2D, BCType.DIRICHLET, 0.0)
            self.mysolv.set_bc(BoundaryAttribute.TOP_2D, BCType.DIRICHLET, 0.0)
            self.mysolv.set_bc(BoundaryAttribute.BOTTOM_2D, BCType.DIRICHLET, 0.0)
            print("  Domaine MOITIE_O : Symétrie centrale activée")

        # Remplissage des coefficients
        Ny, Nx = self.maillage.shape
        print(f"\n=== REMPLISSAGE DES COEFFICIENTS ({Ny}×{Nx}) ===")
        
        for i in range(Ny):
            for j in range(Nx):
                fuel_key = self.maillage[i, j].strip()
                mat = getattr(self, fuel_key) if hasattr(self, fuel_key) and fuel_key else self.R0
                
                for g in range(self.num_groups):
                    self.mysolv.get_D()[g, i, j] = mat['D'][g]
                    self.mysolv.get_NSF()[g, i, j] = mat['NSF'][g]
                    self.mysolv.get_Chi()[g, i, j] = mat['CHI'][g]
                    self.mysolv.get_SigR()[g, i, j] = mat['SIGR'][g]
                
                self.mysolv.get_SigS()[1, 0, i, j] = mat['S12']
                self.mysolv.get_SigS()[0, 1, i, j] = mat['S21']
        
        # Validation
        print("\n=== VALIDATION DES COEFFICIENTS ===")
        for g in range(self.num_groups):
            D = self.mysolv.get_D()[g]
            print(f"Groupe {g}: D ∈ [{D.min():.6f}, {D.max():.6f}] cm")
        print(f"Scattering 0→1: [{self.mysolv.get_SigS()[1, 0].min():.6f}, {self.mysolv.get_SigS()[1, 0].max():.6f}]")
        print(f"Scattering 1→0: [{self.mysolv.get_SigS()[0, 1].min():.6f}, {self.mysolv.get_SigS()[0, 1].max():.6f}]")
        
        # Construction
        self.mysolv.build_matrices()

        time1 = time.time()
        print(f"\n✅ Solveur initialisé en {time1-timeref:.3f} s")

    def solve(self, solver_type=LinearSolverType.GMRES, forward = True, adjoint = False, direct_adjoint = False):
        """Résout le problème avec le solveur spécifié"""
        if self.mysolv is None:
            raise RuntimeError("Le solveur doit être initialisé")

        self.mysolv.set_tolerances(1e-5, 1e-4, 1e-4, 200, 200)
        self.mysolv.set_condensedform(True)
        
        timeref = time.time()

        print("\n=== RÉSOLUTION K-EFF ===")
        if forward :
             self.keff = self.mysolv.SolveKeff(solver_type, True, (round(self.nmeshes/2), round(self.nmeshes/2), 1), 100, 1e-4);

        if direct_adjoint :
             self.keff = self.mysolv.solve_adjoint(solver_type=solver_type, normalize_to_direct=True, use_direct_keff=True)

        if adjoint :
             self.keff = self.mysolv.solve_adjoint(solver_type=solver_type, normalize_to_direct=True, use_direct_keff=False)
        
        if self.order == 0:
            self.phi = np.array([
                self.mysolv.get_flux()[g] for g in range(self.num_groups)
            ])

        if adjoint or direct_adjoint :
            self.phi_adj = np.array([
                self.mysolv.get_flux_adj()[g] for g in range(self.num_groups)
            ])
        
        time1 = time.time()
        
        ecart_pcm = 1E5 * (1/self.kref - 1/self.keff)
        ecart_rel = 100 * (self.keff - self.kref) / self.kref
        
        print("\n" + "="*60)
        print(f"✅ CONVERGENCE ATTEINTE")
        print(f"   k-eff calculé  = {self.keff:.6f}")
        print(f"   k-eff référence = {self.kref:.6f}")
        print(f"   Écart absolu    = {ecart_pcm:+.2f} pcm")
        print(f"   Écart relatif   = {ecart_rel:+.4f} %")
        print(f"   Temps résolution = {time1-timeref:.2f} s")
        print("="*60)

        # Puissance
        if self.order == 0 and self.phi is not None:
            Ny, Nx = self.maillage.shape
            self.pvol = np.zeros((Ny, Nx))
            
            for i in range(Ny):
                for j in range(Nx):
                    for g in range(self.num_groups):
                        nsf_g = self.mysolv.get_NSF()[g]
                        phi_g = self.mysolv.get_flux()[g]
                        self.pvol[i, j] += nsf_g[i, j] * phi_g[i, j]

    def plot_flux(self, group=0, fine=False):
        """Visualise le flux"""

        if self.phi is None:
            print("❌ Flux non disponible")
            return

        if fine : 
            flux = self.phi_fins
        else : 
            flux = self.phi
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(flux[group], cmap='jet', 
                   cbar_kws={'label': f'Flux φ{group+1}'})
        plt.title(f"Flux Groupe {group+1} - k-eff = {self.keff:.5f}")
        plt.tight_layout()
        plt.show()

    def plot_pvol(self):
        """Visualise la puissance"""
        if self.pvol is None:
            print("❌ Puissance non disponible")
            return
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(self.pvol, cmap='jet', 
                   cbar_kws={'label': 'Puissance'})
        plt.title(f"Distribution de puissance - k-eff = {self.keff:.5f}")
        plt.tight_layout()
        plt.show()


# ===== SCRIPT PRINCIPAL =====
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(
        description="IAEA 2D - Solveur neutronique avancé"
    )
    parser.add_argument("--mesh", type=str, default="9x9", 
                       help="Résolution du maillage")
    parser.add_argument("--domain", type=str, default="entier",
                       choices=["entier", "quart_so", "moitie_s", "moitie_o"],
                       help="Géométrie du domaine")
    parser.add_argument("--solver", type=str, default="gmres",
                       choices=["bicgstab", "gmres", "minres", "cg", "pcg", "fgmres"],
                       help="Solveur linéaire")
    parser.add_argument("--upscatter", action="store_true",
                       help="Activer l'upscattering", default= False)
    parser.add_argument("--adjoint", action="store_true",
                       help="Activer le calcul de l'adjoint", default= False)
    parser.add_argument("--adjoint_from_forward", action="store_true",
                       help="Activer le calcul de l'adjoint en utilisant les résultat du forward", default= False)
    parser.add_argument("--plot", action="store_true",
                       help="Afficher les graphiques", default= False)
    parser.add_argument("--ordre", type=int, default=0,
                       help="Ordre des élements RT")
    parser.add_argument("--harmonics", type=int, default=0,
                       help="Nb d'harmoniques a calculer")
    args = parser.parse_args()
    
    # Conversion du nom du solveur
    solver_map = {
        "bicgstab": LinearSolverType.BICGSTAB,
        "gmres": LinearSolverType.GMRES,
        "minres": LinearSolverType.MINRES,
        "fgmres": LinearSolverType.FGMRES,
        "pcg": LinearSolverType.PCG,
        "cg": LinearSolverType.CG
    }
    
    print("="*60)
    print("BENCHMARK IAEA 2D")
    print("="*60)
    print(f"Maillage : {args.mesh}")
    print(f"Domaine  : {args.domain}")
    print(f"Solveur  : {args.solver}")
    print(f"Upscatter: {'Oui' if args.upscatter else 'Non'}")
    print("="*60)
    
    # Création
    iaea2d = Iaea2D(meshtype=args.mesh, domaine=args.domain)
    iaea2d.order = args.ordre
    iaea2d.load_iaea2d_mat(include_upscattering=args.upscatter)
    iaea2d.mesh_initialisation()
    
    # Résolution avec le solveur spécifié
    iaea2d.init_solver()
    iaea2d.solve(solver_type=solver_map[args.solver])
    iaea2d.phi_fins = iaea2d.mysolv.zoom((2,2,1))
    
    if args.adjoint_from_forward:
    	 # Problème adjoint
    	 iaea2d.solve(solver_type=solver_map[args.solver], adjoint=False, direct_adjoint=True)
    
    if args.adjoint:
    	 # Problème adjoint sans utiliser le forward
    	 iaea2d.solve(solver_type=solver_map[args.solver], adjoint=True, direct_adjoint=False)
    

    # Export
    iaea2d.mysolv.save_vtk("iaea2d_result")
    iaea2d.mysolv.save_vtk_coarse("iaea2d_result_coarse")
    iaea2d.mysolv.save_vtk_zoom("iaea2d_result_zoom")
    print("\n📁 Résultats exportés: iaea2d_result.vtk")
    
    # Visualisation
    if args.plot:
        iaea2d.plot_flux(group=0)
        iaea2d.plot_flux(group=1)
        iaea2d.plot_pvol()

        iaea2d.plot_flux(group=0, fine=True)
        iaea2d.plot_flux(group=1, fine=True)

    if args.harmonics > 0:
        # Calculer les harmoniques
        alphas = iaea2d.mysolv.solve_harmonics(n_harmonics=args.harmonics)
        print(f"Valeurs propres harmoniques: {alphas}")
        iaea2d.mysolv.save_harmonics_vtk("harmonics")

    print(f"\n⏱️  Temps total : {time.time() - iaea2d.start:.2f} s")
    print("="*60)
