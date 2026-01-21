# NeutMFEM - Solveur de Diffusion Neutronique Multi-Groupes

> ⚠️ **Version Alpha** : Ce projet est en cours de développement et de test. L'API peut évoluer et certaines fonctionnalités peuvent être incomplètes ou instables.

## Description

NeutMFEM est un solveur haute performance pour l'équation de diffusion neutronique multi-groupes utilisant la méthode des éléments finis mixtes (Raviart-Thomas / L2 discontinu).

## Fonctionnalités

### Résolution
- **Problème aux valeurs propres** : Calcul de k-eff par itérations de puissance avec accélération de Chebyshev
- **Modes propres (harmoniques)** : Calcul des N premiers modes propres avec déflation de Hotelling
- **Problème adjoint** : Calcul du flux adjoint pour analyses de sensibilité
- **Calculs sous-critiques** : Résolution avec sources externes fixes
- **Raffinement (zoom)** : Amélioration de la résolution spatiale sans recalcul complet

### Conditions aux limites
- **Dirichlet** : Flux imposé (φ = value)
- **Neumann** : Courant imposé (J·n = value)
- **Robin** : Condition mixte (α·φ + β·J·n = value)
- **Mirror** : Réflexion spéculaire (J·n = 0)

### Géométrie
- Support 1D, 2D et 3D
- Maillages cartésiens structurés
- Symétries (quart rotationnelle, centrale)
- Modélisation des réflecteurs

## Architecture

```
NeutMFEM/
├── docs/                       # Documentation et ressources
│   ├── logo.png                #   Logo (thème sombre)
│   └── logo_light.png          #   Logo (thème clair)
├── include/                    # Headers C++
│   ├── NeutMFEM.hpp            #   Définition de la classe principale
│   └── solver.hpp              #   Algorithmes d'accélération
├── src/                        # Sources C++
│   ├── NeutMFEM.cpp            #   Implémentation du solveur
│   ├── solver.cpp              #   Implémentation des accélérateurs
│   └── wrapper.cpp             #   Bindings Python (pybind11)
├── lib/                        # Fichiers objets compilés (.o)
├── neutmfem/                   # Module Python
│   └── _neutmfem.so            #   Bibliothèque partagée
├── tests/                      # Tests et benchmarks Python
│   ├── biblis2d/
│   │   └── biblis2D.py         #   Benchmark Biblis 2D
│   ├── iaea2d/
│   │   └── iaea2d.py           #   Benchmark IAEA 2D
│   └── iaea3d/
│       └── iaea3d.py           #   Benchmark IAEA 3D
├── Makefile                    # Script de compilation
└── README.md                   # Cette documentation
```

## Dépendances

Le projet a été développé et testé avec les versions suivantes :

| Composant | Version |
|-----------|---------|
| Python | 3.13 |
| GCC | 13.2.0 |
| pybind11 | 2.13.6 |
| MFEM | via Anaconda |

## Installation

### Prérequis

1. **Environnement Anaconda avec MFEM** :
```bash
conda create -n mfem python=3.13
conda activate mfem
conda install -c conda-forge mfem
```

2. **pybind11 2.13.6** : Télécharger depuis [github.com/pybind/pybind11](https://github.com/pybind/pybind11)

3. **GCC 13.2.0** : Compilateur C++17

### Configuration du Makefile

Éditer le `Makefile` pour adapter les chemins à votre environnement :

```makefile
# Chemin vers GCC
GCC = /path/to/gcc-13.2.0/install

# Chemin vers l'environnement Anaconda avec MFEM
ANACONDA_VERSION = /path/to/anaconda3/envs/mfem

# Chemin vers pybind11
PYBIND = /path/to/pybind11-2.13.6
```

### Compilation

```bash
# Nettoyage et compilation complète
make clean
make all -j4
```

Le module Python `_neutmfem.so` sera généré dans le répertoire `neutmfem/`.

## Utilisation

### Import et énumérations

```python
import neutmfem._neutmfem as NeutMFEM
from neutmfem._neutmfem import (
    BCType,              # DIRICHLET, NEUMANN, ROBIN, MIRROR
    BoundaryAttribute,   # LEFT_2D, RIGHT_2D, TOP_2D, BOTTOM_2D, etc.
    LinearSolverType,    # BICGSTAB, GMRES, MINRES, CG, PCG, FGMRES
    VerbosityLevel       # SILENT, LIGHT, NORMAL, DEBUG
)
```

### Exemple de base

```python
import numpy as np
import neutmfem._neutmfem as NeutMFEM
from neutmfem._neutmfem import BCType, BoundaryAttribute, LinearSolverType

# Création d'un solveur 2D avec 2 groupes d'énergie
# Paramètres : (order, num_groups, x_breaks, y_breaks, z_breaks)
x_breaks = np.linspace(0, 100, 11).tolist()  # 10 cellules en X
y_breaks = np.linspace(0, 100, 11).tolist()  # 10 cellules en Y
z_breaks = [0.0]  # 2D (z_breaks avec un seul élément)

solver = NeutMFEM.NeutMFEM(0, 2, x_breaks, y_breaks, z_breaks)

# Configuration des propriétés matériaux (accès zero-copy)
D = solver.get_D()          # shape: (num_groups, ny, nx)
D[0, :, :] = 1.5            # Groupe rapide
D[1, :, :] = 0.5            # Groupe thermique

SigR = solver.get_SigR()    # Section efficace de removal
SigR[0, :, :] = 0.025
SigR[1, :, :] = 0.010

NSF = solver.get_NSF()      # ν·Σf (production de neutrons)
NSF[0, :, :] = 0.020
NSF[1, :, :] = 0.005

Chi = solver.get_Chi()      # Spectre de fission
Chi[0, :, :] = 1.0          # Tous les neutrons naissent dans le groupe rapide
Chi[1, :, :] = 0.0

SigS = solver.get_SigS()    # shape: (g_to, g_from, ny, nx)
SigS[1, 0, :, :] = 0.017    # Scattering 0→1 (down-scattering)

# Conditions aux limites
solver.set_bc(BoundaryAttribute.LEFT_2D, BCType.MIRROR)
solver.set_bc(BoundaryAttribute.BOTTOM_2D, BCType.MIRROR)
solver.set_bc(BoundaryAttribute.RIGHT_2D, BCType.DIRICHLET, 0.0)
solver.set_bc(BoundaryAttribute.TOP_2D, BCType.DIRICHLET, 0.0)

# Construction des matrices et résolution
solver.build_matrices()
keff = solver.SolveKeff(LinearSolverType.GMRES)
print(f"k-eff = {keff:.6f}")

# Récupération du flux
flux = solver.get_flux()  # shape: (num_groups, ny, nx)
print(f"Flux shape: {flux.shape}")

# Export VTK
solver.save_vtk("solution")
```

### Configuration avancée

```python
# Paramètres de convergence
solver.set_tolerances(
    tol_keff=1e-5,      # Tolérance sur k-eff
    tol_flux=1e-5,      # Tolérance sur le flux
    tol_linear=1e-5,    # Tolérance du solveur linéaire
    max_outer=200,      # Itérations externes max
    max_inner=200       # Itérations internes max
)

# Dimension de Krylov (pour GMRES)
solver.set_krylov_dimension(100)

# Niveau de verbosité
solver.set_verbosity(VerbosityLevel.DEBUG)

# Activer la forme condensée (recommandé pour la performance)
solver.set_condensedform(True)

# Résolution avec initialisation coarse
# Les facteurs coarse peuvent être passés comme tuple (fx, fy, fz)
keff = solver.SolveKeff(
    LinearSolverType.GMRES,
    use_coarse_init=True,
    coarse_factors=(17, 17, 1),  # Facteurs d'homogénéisation
    coarse_max_iter=100,
    coarse_tol_keff=1e-4
)
```

### Symétries

> ⚠️ **Bug connu** : Un bug a été détecté sur les symétries. Les résultats ne sont pas garantis lors de l'utilisation de `apply_quarter_rotational_symmetry()` ou `apply_central_symmetry()`.

```python
# Symétrie quart rotationnelle (axes X et Y)
solver.apply_quarter_rotational_symmetry(axis1=0, axis2=1)

# Symétrie centrale
solver.apply_central_symmetry(axis1=0, axis2=1)
```

### Problème adjoint

```python
# Résolution du problème adjoint
keff_adj = solver.solve_adjoint(
    solver_type=LinearSolverType.GMRES,
    normalize_to_direct=True,   # Normaliser par rapport au flux direct
    use_direct_keff=True        # Utiliser k-eff du calcul direct
)

# Récupération du flux adjoint
flux_adj = solver.get_flux_adj()

# Importance d'un groupe
importance_g0 = solver.get_group_importance(0)
```

### Calcul des modes propres (harmoniques)

```python
from neutmfem._neutmfem import EigensolverParameters

# Configuration optionnelle des paramètres
params = EigensolverParameters(
    tol_eigenvalue=1e-5,
    tol_eigenvector=1e-4,
    max_iter_per_mode=500,
    deflation_weight=1.0,
    orthogonalize=True,
    normalize_eigenvectors=True
)

# Calcul des 5 premiers modes propres
modes = solver.solve_eigenmodes(5, LinearSolverType.GMRES, params)

# Afficher les résultats
for m in modes:
    print(f"Mode {m.mode_index}: λ = {m.lambda_:.8f}, k = {m.keff:.6f}")
    print(f"  Ratio de dominance: {m.dominance_ratio:.4f}")
    print(f"  Convergé: {m.converged}, Itérations: {m.iterations}")

# Récupérer les eigenvectors
phi_0 = solver.get_eigenvector(0)  # Mode fondamental
phi_1 = solver.get_eigenvector(1)  # Première harmonique

# Récupérer toutes les eigenvalues
eigenvalues = solver.get_eigenvalues()  # [λ₀, λ₁, λ₂, ...]
keff_values = solver.get_keff_values()   # [k₀, k₁, k₂, ...]

# Vérifier l'orthogonalité des modes
ortho_matrix = solver.check_orthogonality()

# Export VTK des eigenvectors
solver.save_eigenvectors_vtk("harmonics", max_modes=5)

# Modes propres adjoints
adj_modes = solver.solve_adjoint_eigenmodes(3, LinearSolverType.GMRES, params)
phi_adj_1 = solver.get_adjoint_eigenvector(1)
```

### Réflecteurs

```python
import numpy as np

# Définition d'un réflecteur
D_refl = np.array([1.32, 0.2772])
SigR_refl = np.array([0.0257, 0.0716])
SigS_refl = np.array([[0.0, 0.0231], [0.0, 0.0]])  # shape: (g_to, g_from)

refl_id = solver.add_refl(D_refl, SigR_refl, SigS_refl)

# Activation sur une frontière
# dimension: 0=X, 1=Y, 2=Z
# is_upper: True=max, False=min
solver.set_refl(refl_id, dimension=0, is_upper=True)  # X max

# Nettoyage des réflecteurs
solver.clean_refl()
```

### Raffinement (Zoom)

```python
# Résolution sur maillage grossier
keff = solver.SolveKeff(LinearSolverType.GMRES)

# Raffinement 2×2 avec sources figées
# Paramètres : (refine_factors, adjoint, solver_type)
flux_fine = solver.zoom((2, 2, 1), adjoint=False, solver_type=LinearSolverType.GMRES)
print(f"Flux raffiné shape: {flux_fine.shape}")

# Export VTK du maillage zoomé
solver.save_vtk_zoom("solution_zoom")
```

### Conditions aux limites Robin

```python
# Configuration des coefficients Robin : α·φ + β·J·n = value
solver.set_bc(BoundaryAttribute.RIGHT_2D, BCType.ROBIN, value=0.0)
solver.set_robin_coefficients(BoundaryAttribute.RIGHT_2D, alpha=1.0, beta=0.5)
```

### Export VTK

```python
# Export principal (flux, sections efficaces)
solver.save_vtk("result", coarse=True, zoom=True)

# Export du maillage coarse uniquement
solver.save_vtk_coarse("result_coarse")

# Export du maillage zoomé uniquement
solver.save_vtk_zoom("result_zoom")
```

## Référence API

### Accesseurs de données (zero-copy)

| Méthode | Shape | Description |
|---------|-------|-------------|
| `get_D()` | (G, ny, nx) | Coefficient de diffusion |
| `get_SigR()` | (G, ny, nx) | Section efficace de removal |
| `get_NSF()` | (G, ny, nx) | ν·Σf (production) |
| `get_KSF()` | (G, ny, nx) | κ·Σf (énergie) |
| `get_Chi()` | (G, ny, nx) | Spectre de fission |
| `get_SigS()` | (G, G, ny, nx) | Matrice de scattering |
| `get_src()` | (G, ny, nx) | Sources externes |
| `get_flux()` | (G, ny, nx) | Flux scalaire |
| `get_flux_adj()` | (G, ny, nx) | Flux adjoint |

### Énumérations

**BoundaryAttribute (2D):**
- `LEFT_2D`, `RIGHT_2D`, `TOP_2D`, `BOTTOM_2D`

**BoundaryAttribute (3D):**
- `FRONT_3D`, `BACK_3D`, `LEFT_3D`, `RIGHT_3D`, `TOP_3D`, `BOTTOM_3D`

**BCType:**
- `DIRICHLET`, `NEUMANN`, `ROBIN`, `MIRROR`

**LinearSolverType:**
- `BICGSTAB`, `GMRES`, `MINRES`, `CG`, `PCG`, `FGMRES`

**VerbosityLevel:**
- `SILENT`, `LIGHT`, `NORMAL`, `DEBUG`

## Formulation Mathématique

### Équation de diffusion multi-groupes

Pour chaque groupe d'énergie g :

$$-\nabla \cdot D_g \nabla \phi_g + \Sigma_{r,g} \phi_g = \frac{\chi_g}{k_{eff}} \sum_{g'} \nu\Sigma_{f,g'} \phi_{g'} + \sum_{g' \neq g} \Sigma_{s,g' \to g} \phi_{g'}$$

### Formulation mixte

Le système est reformulé en utilisant le courant J = -D∇φ :

$$\frac{1}{D_g} \mathbf{J}_g + \nabla \phi_g = 0$$
$$-\nabla \cdot \mathbf{J}_g + \Sigma_{r,g} \phi_g = S_g$$

### Forme condensée

Pour améliorer les performances, le système peut être condensé :

$$K_g \cdot \Phi_g = S_g$$

où $K_g = B \cdot A_g^{-1} \cdot B^T + C_g$

### Discrétisation

- **Courant J** : Éléments de Raviart-Thomas RT_k
- **Flux φ** : Éléments L2 discontinus P_k (DG)
- **Coefficients** : L2 constant par élément

### Ordres d'éléments finis disponibles

Le paramètre `order` du constructeur permet de choisir l'ordre des éléments finis :

| Order | Courant J | Flux φ | DOFs/élément (2D) | Précision spatiale |
|-------|-----------|--------|-------------------|-------------------|
| 0 | RT₀ | P₀ | 4 + 1 = 5 | O(h) |
| 1 | RT₁ | P₁ | 12 + 3 = 15 | O(h²) |
| 2 | RT₂ | P₂ | 24 + 6 = 30 | O(h³) |

**Éléments de Raviart-Thomas RT_k :**

Les espaces RT_k sont des espaces vectoriels H(div)-conformes définis sur chaque élément K :

$$RT_k(K) = (P_k)^d + \mathbf{x} \cdot P_k$$

où d est la dimension spatiale. Ces éléments garantissent la continuité de la composante normale du courant à travers les faces des éléments, ce qui assure la conservation locale des neutrons.

- **RT₀** : Courant linéaire par composante, 1 DOF par face (normale constante)
- **RT₁** : Courant quadratique, 2 DOFs par face + DOFs internes
- **RT₂** : Courant cubique, 3 DOFs par face + DOFs internes

**Éléments L2 discontinus P_k :**

Les espaces P_k sont des polynômes de degré ≤ k, discontinus entre éléments :

- **P₀** : Flux constant par élément (1 DOF)
- **P₁** : Flux linéaire (3 DOFs en 2D, 4 en 3D)
- **P₂** : Flux quadratique (6 DOFs en 2D, 10 en 3D)

**Compatibilité RT_k / P_k :**

La formulation mixte requiert une condition inf-sup (LBB) satisfaite par le couple RT_k / P_k. L'opérateur divergence appliqué à RT_k donne exactement P_k :

$$\nabla \cdot RT_k = P_k$$

Cette compatibilité garantit la stabilité du schéma sans nécessiter de stabilisation artificielle.

> ⚠️ **Limitation actuelle** : À ce jour, seul l'ordre 0 (RT₀-P₀) est pleinement fonctionnel et testé. Les ordres supérieurs sont en cours de développement.

## Algorithmes

### Itérations de puissance
1. Initialisation (optionnelle sur maillage grossier)
2. Calcul de la source de fission totale
3. Résolution groupe par groupe (down-scattering)
4. Mise à jour de k-eff
5. Accélération de Chebyshev
6. Test de convergence

### Accélération de Chebyshev

Polynômes de Chebyshev pour accélérer la convergence des itérations de puissance. Particulièrement efficace lorsque le ratio de dominance spectrale est proche de 1.

### Accélération d'Anderson

Alternative basée sur les moindres carrés pour les cas difficiles. Utilise une décomposition de Cholesky pour résoudre le système normal.

### Calcul des modes propres

Itérations de puissance inverse avec déflation de Hotelling pour calculer les modes supérieurs. Les modes sont orthogonalisés avec le produit scalaire F-pondéré.

## Benchmarks supportés

- IAEA 2D/3D
- Biblis 2D
