"""
BENCHMARK BIBLIS 2D - Version améliorée
Nouvelles fonctionnalités :
- Solveurs linéaires configurables
- Symétries automatiques selon le domaine
- Upscattering optionnel
"""

import time
import numpy as np
import argparse

import neutmfem._neutmfem as neutron_solver
from neutmfem._neutmfem import BCType, BoundaryAttribute, LinearSolverType

import seaborn as sns
import matplotlib.pyplot as plt

np.set_printoptions(threshold=np.inf, linewidth=np.inf)


class Biblis2D:
    """
    Benchmark BIBLIS 2D avec fonctionnalités avancées
    """
    
    def __init__(self, meshtype="2x2", domaine="entier", ncpu=1, sym="cyclique"):
        
        self.start = time.time()
        self.meshtype = meshtype
        self.domaine = domaine
        self.ncpu = ncpu
        self.sym = sym
        
        self.kref = 1.02511  # k-eff référence IAEA
        self.num_groups = 2
        self.verbose = 0
        self.order_phi = 0
        self.order_J = 0
        
        self.init_meshing = False
        self.mysolv = None
        self.keff = None
        self.phi = None
        self.pvol = None

        # Maillage 17×17 assemblages
        self.maillage_motifs_coeur = np.array([
            ["  ", "  ", "  ", "  ", "  ", "  ", "  ", "  ", "  ", "  ", "  ", "  ", "  ", "  ", "  ", "  ", "  "],
            ["  ", "  ", "  ", "  ", "  ", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "  ", "  ", "  ", "  ", "  "],
            ["  ", "  ", "  ", "F4", "F4", "F8", "F1", "F1", "F1", "F1", "F1", "F8", "F4", "F4", "  ", "  ", "  "],
            ["  ", "  ", "F4", "F4", "F5", "F1", "F7", "F1", "F7", "F1", "F7", "F1", "F5", "F4", "F4", "  ", "  "],
            ["  ", "  ", "F4", "F5", "F2", "F8", "F2", "F8", "F1", "F8", "F2", "F8", "F2", "F5", "F4", "  ", "  "],
            ["  ", "F4", "F8", "F1", "F8", "F2", "F8", "F2", "F6", "F2", "F8", "F2", "F8", "F1", "F8", "F4", "  "],
            ["  ", "F4", "F1", "F7", "F2", "F8", "F1", "F8", "F2", "F8", "F1", "F8", "F2", "F7", "F1", "F4", "  "],
            ["  ", "F4", "F1", "F1", "F8", "F2", "F8", "F1", "F8", "F1", "F8", "F2", "F8", "F1", "F1", "F4", "  "],
            ["  ", "F4", "F1", "F7", "F1", "F6", "F2", "F8", "F1", "F8", "F2", "F6", "F1", "F7", "F1", "F4", "  "],
            ["  ", "F4", "F1", "F1", "F8", "F2", "F8", "F1", "F8", "F1", "F8", "F2", "F8", "F1", "F1", "F4", "  "],
            ["  ", "F4", "F1", "F7", "F2", "F8", "F1", "F8", "F2", "F8", "F1", "F8", "F2", "F7", "F1", "F4", "  "],
            ["  ", "F4", "F8", "F1", "F8", "F2", "F8", "F2", "F6", "F2", "F8", "F2", "F8", "F1", "F8", "F4", "  "],
            ["  ", "  ", "F4", "F5", "F2", "F8", "F2", "F8", "F1", "F8", "F2", "F8", "F2", "F5", "F4", "  ", "  "],
            ["  ", "  ", "F4", "F4", "F5", "F1", "F7", "F1", "F7", "F1", "F7", "F1", "F5", "F4", "F4", "  ", "  "],
            ["  ", "  ", "  ", "F4", "F4", "F8", "F1", "F1", "F1", "F1", "F1", "F8", "F4", "F4", "  ", "  ", "  "],
            ["  ", "  ", "  ", "  ", "  ", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "  ", "  ", "  ", "  ", "  "],
            ["  ", "  ", "  ", "  ", "  ", "  ", "  ", "  ", "  ", "  ", "  ", "  ", "  ", "  ", "  ", "  ", "  "]
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
        plt.title("Distribution des matériaux Biblis 2D")
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
        cell_size = 23.1226 / self.nmeshes
        nx_cells = self.maillage.shape[1]
        ny_cells = self.maillage.shape[0]

        self.x_breaks = np.linspace(0.0, nx_cells * cell_size, nx_cells + 1)
        self.y_breaks = np.linspace(0.0, ny_cells * cell_size, ny_cells + 1)
        self.z_breaks = np.array([0.0])
        
        time1 = time.time()
        print(f"✅ Maillage initialisé: {ny_cells}×{nx_cells} cellules "
              f"({time1-timeref:.3f} s)")
        self.init_meshing = True

    def load_biblis2d_mat(self, include_upscattering=False):
        """Charge les matériaux avec upscattering optionnel"""
        
        # Matériaux F1-F8
        self.F1 = {
            'D': [1.4360, 0.3635],
            'ABS': [0.0095042, 0.0750580],
            'NSF': [0.0058708, 0.0960670],
            'CHI': [1., 0.],
            'S12': 0.017754
        }
        self.F1['SIGR'] = [self.F1['ABS'][0] + self.F1['S12'], self.F1['ABS'][1]]

        self.F2 = {
            'D': [1.4366, 0.3636],
            'ABS': [0.0096785, 0.0784360],
            'NSF': [0.0061908, 0.1035800],
            'CHI': [1., 0.],
            'S12': 0.017621
        }
        self.F2['SIGR'] = [self.F2['ABS'][0] + self.F2['S12'], self.F2['ABS'][1]]

        self.F4 = {
            'D': [1.4389, 0.3638],
            'ABS': [0.0103630, 0.0914080],
            'NSF': [0.0074527, 0.1323600],
            'CHI': [1., 0.],
            'S12': 0.017101
        }
        self.F4['SIGR'] = [self.F4['ABS'][0] + self.F4['S12'], self.F4['ABS'][1]]

        self.F5 = {
            'D': [1.4381, 0.3665],
            'ABS': [0.0100030, 0.0848280],
            'NSF': [0.0061908, 0.1035800],
            'CHI': [1., 0.],
            'S12': 0.01729
        }
        self.F5['SIGR'] = [self.F5['ABS'][0] + self.F5['S12'], self.F5['ABS'][1]]

        self.F6 = {
            'D': [1.4385, 0.3665],
            'ABS': [0.0101320, 0.0873140],
            'NSF': [0.0064285, 0.1091100],
            'CHI': [1., 0.],
            'S12': 0.017192
        }
        self.F6['SIGR'] = [self.F6['ABS'][0] + self.F6['S12'], self.F6['ABS'][1]]

        self.F7 = {
            'D': [1.4389, 0.3679],
            'ABS': [0.0101650, 0.0880240],
            'NSF': [0.0061908, 0.1035800],
            'CHI': [1., 0.],
            'S12': 0.017125
        }
        self.F7['SIGR'] = [self.F7['ABS'][0] + self.F7['S12'], self.F7['ABS'][1]]

        self.F8 = {
            'D': [1.4393, 0.3680],
            'ABS': [0.0102940, 0.0905100],
            'NSF': [0.0064285, 0.1091100],
            'CHI': [1., 0.],
            'S12': 0.017027
        }
        self.F8['SIGR'] = [self.F8['ABS'][0] + self.F8['S12'], self.F8['ABS'][1]]

        # Réflecteur
        self.R0 = {
            'D': [1.3200, 0.2772],
            'ABS': [0.0026562, 0.0715960],
            'NSF': [0.0000000, 0.0000000],
            'CHI': [0., 0.],
            'S12': 0.023106
        }
        self.R0['SIGR'] = [self.R0['ABS'][0] + self.R0['S12'], self.R0['ABS'][1]]
        
        # ✅ NOUVEAU : Upscattering optionnel
        if include_upscattering:
            print("\n⚠️  ATTENTION : Ajout d'upscattering (NON standard)")
            upscatter_ratio = 0.08
            
            for fuel_name in ['F1', 'F2', 'F4', 'F5', 'F6', 'F7', 'F8', 'R0']:
                fuel = getattr(self, fuel_name)
                fuel['S21'] = fuel['S12'] * upscatter_ratio
                fuel['SIGR'][1] = fuel['ABS'][1] + fuel['S21']
                print(f"  {fuel_name}: Σₛ(1→0)={fuel['S21']:.6f}")
        else:
            for fuel_name in ['F1', 'F2', 'F4', 'F5', 'F6', 'F7', 'F8', 'R0']:
                getattr(self, fuel_name)['S21'] = 0.0
        
        print("✅ Matériaux chargés: F1-F8 + R0")

    def init_solver(self):
        """Initialise le solveur avec configuration avancée"""
        if not self.init_meshing:
            raise RuntimeError("Le maillage doit être initialisé avant le solveur")
        
        timeref = time.time()

        # Instanciation
        self.mysolv = neutron_solver.NeutMFEM(
            self.order_phi,
	    self.order_J,
	    self.num_groups,
            self.x_breaks.tolist(),
            self.y_breaks.tolist(),
            self.z_breaks.tolist()
        )
        
        # ✅ Configuration optionnelle de Krylov dim pour GMRES
        # Peut être ajustée avec solver.set_krylov_dimension(100) si besoin
        
        # ✅ Configuration automatique des symétries
        print("\n=== CONFIGURATION DES CONDITIONS AUX LIMITES ===")
        
        if self.domaine == "entier":
            # Vacuum sur tous les bords
            for attr in [BoundaryAttribute.BOTTOM_2D, BoundaryAttribute.RIGHT_2D,
                        BoundaryAttribute.TOP_2D, BoundaryAttribute.LEFT_2D]:
                self.mysolv.set_bc(attr, BCType.DIRICHLET, 0.0)
            print("  Domaine ENTIER : Vacuum sur tous les bords")
        
        elif self.domaine == "quart_so":
            # ✅ Symétrie quart cyclique automatique
            self.mysolv.apply_quarter_rotational_symmetry(0, 1)
            self.mysolv.set_bc(BoundaryAttribute.RIGHT_2D, BCType.DIRICHLET, 0.0)
            self.mysolv.set_bc(BoundaryAttribute.TOP_2D, BCType.DIRICHLET, 0.0)
            print("  Domaine QUART_SO : Symétrie cyclique activée")
        
        elif self.domaine == "moitie_s":
            # ✅ Symétrie centrale selon Y (moitié Sud)
            self.mysolv.apply_central_symmetry(0, 1)
            self.mysolv.set_bc(BoundaryAttribute.TOP_2D, BCType.MIRROR)
            self.mysolv.set_bc(BoundaryAttribute.BOTTOM_2D, BCType.DIRICHLET, 0.0)
            self.mysolv.set_bc(BoundaryAttribute.RIGHT_2D, BCType.DIRICHLET, 0.0)
            self.mysolv.set_bc(BoundaryAttribute.LEFT_2D, BCType.DIRICHLET, 0.0)
            print("  Domaine MOITIE_S : Symétrie centrale activée")
        
        elif self.domaine == "moitie_o":
            # ✅ Symétrie centrale selon X (moitié Ouest)
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
        
        if self.order_phi == 0:
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
        if self.order_phi == 0 and self.phi is not None:
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
        description="BIBLIS 2D - Solveur neutronique avancé"
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
    parser.add_argument("--ordre_phi", type=int, default=0,
                       help="Ordre des élements du flux")
    parser.add_argument("--ordre_J", type=int, default=-1,
                       help="Ordre des élements RT du courant")
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
    print("BENCHMARK BIBLIS 2D - Version Avancée")
    print("="*60)
    print(f"Maillage : {args.mesh}")
    print(f"Domaine  : {args.domain}")
    print(f"Solveur  : {args.solver}")
    print(f"Upscatter: {'Oui' if args.upscatter else 'Non'}")
    print("="*60)
    
    # Création
    biblis2d = Biblis2D(meshtype=args.mesh, domaine=args.domain)
    biblis2d.order_phi = args.ordre_phi
    
    if args.ordre_J != -1 :
        biblis2d.order_J = args.ordre_J
    else :
        biblis2d.order_J = biblis2d.order_phi

    biblis2d.load_biblis2d_mat(include_upscattering=args.upscatter)
    biblis2d.mesh_initialisation()
    
    # Résolution avec le solveur spécifié
    biblis2d.init_solver()
    biblis2d.solve(solver_type=solver_map[args.solver])
    biblis2d.phi_fins = biblis2d.mysolv.zoom((2,2,1))
    
    if args.adjoint_from_forward:
         # Problème adjoint
         biblis2d.solve(solver_type=solver_map[args.solver], adjoint=False, direct_adjoint=True)
    
    if args.adjoint:
         # Problème adjoint sans utiliser le forward
         biblis2d.solve(solver_type=solver_map[args.solver], adjoint=True, direct_adjoint=False)
    
    # Export
    biblis2d.mysolv.save_vtk("biblis2d_result")
    biblis2d.mysolv.save_vtk_coarse("biblis2d_result_coarse")
    biblis2d.mysolv.save_vtk_zoom("biblis2d_result_zoom")
    print("\n📁 Résultats exportés: biblis2d_result.vtk")
    
    # Visualisation
    if args.plot:
        biblis2d.plot_flux(group=0, fine=False)
        biblis2d.plot_flux(group=1, fine=False)
        biblis2d.plot_pvol()

        biblis2d.plot_flux(group=0, fine=True)
        biblis2d.plot_flux(group=1, fine=True)


    if args.harmonics > 0 :
        # Calculer les 5 premiers harmoniques
        results = biblis2d.mysolv.solve_eigenmodes(args.harmonics)

        # Afficher les résultats
        for r in results:
            print(f"Mode {r.mode_index}: k = {r.eigenvalue:.6f}")
            print(f"  Ratio de dominance: {r.dominance_ratio:.4f}")
            print(f"  Convergé: {r.converged}, Itérations: {r.iterations}")

        # Récupérer les flux
        phi_1 = biblis2d.mysolv.get_eigenvector(1)
        phi_2 = biblis2d.mysolv.get_eigenvector(2)
        phi_3 = biblis2d.mysolv.get_eigenvector(3)
        phi_4 = biblis2d.mysolv.get_eigenvector(4)

        # Récupérer toutes les eigenvalues
        eigenvalues = biblis2d.mysolv.get_eigenvalues()
        print(eigenvalues)

        # Export VTK
        biblis2d.mysolv.save_eigenvectors_vtk("harmonics_output", max_modes=5)

    print(f"\n⏱️  Temps total : {time.time() - biblis2d.start:.2f} s")
    print("="*60)
