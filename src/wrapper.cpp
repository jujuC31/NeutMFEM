/**
 * @file wrapper.cpp
 * @brief Wrapper pybind11 pour NeutMFEM
 */

#include "NeutMFEM.hpp"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;
using namespace neutmfem;

PYBIND11_MODULE(_neutmfem, m) {
    m.doc() = "NeutMFEM - Solveur de diffusion neutronique multi-groupes";

    // Énumérations
    py::enum_<VerbosityLevel>(m, "VerbosityLevel", py::arithmetic())
        .value("SILENT", VerbosityLevel::SILENT)
        .value("LIGHT", VerbosityLevel::LIGHT)
        .value("NORMAL", VerbosityLevel::NORMAL)
        .value("DEBUG", VerbosityLevel::DEBUG)
        .export_values();

    py::enum_<LinearSolverType>(m, "LinearSolverType", py::arithmetic())
        .value("BICGSTAB", LinearSolverType::BICGSTAB)
        .value("GMRES", LinearSolverType::GMRES)
        .value("MINRES", LinearSolverType::MINRES)
        .value("CG", LinearSolverType::CG)
        .value("PCG", LinearSolverType::PCG)
        .value("FGMRES", LinearSolverType::FGMRES)
        .export_values();

    py::enum_<BCType>(m, "BCType", py::arithmetic())
        .value("DIRICHLET", BCType::DIRICHLET)
        .value("NEUMANN", BCType::NEUMANN)
        .value("ROBIN", BCType::ROBIN)
        .value("MIRROR", BCType::MIRROR)
        .export_values();

    py::enum_<BoundaryAttribute>(m, "BoundaryAttribute", py::arithmetic())
        .value("LEFT_1D", BoundaryAttribute::LEFT_1D)
        .value("RIGHT_1D", BoundaryAttribute::RIGHT_1D)
        .value("LEFT_2D", BoundaryAttribute::LEFT_2D)
        .value("RIGHT_2D", BoundaryAttribute::RIGHT_2D)
        .value("TOP_2D", BoundaryAttribute::TOP_2D)
        .value("BOTTOM_2D", BoundaryAttribute::BOTTOM_2D)
        .value("FRONT_3D", BoundaryAttribute::FRONT_3D)
        .value("BACK_3D", BoundaryAttribute::BACK_3D)
        .value("LEFT_3D", BoundaryAttribute::LEFT_3D)
        .value("RIGHT_3D", BoundaryAttribute::RIGHT_3D)
        .value("TOP_3D", BoundaryAttribute::TOP_3D)
        .value("BOTTOM_3D", BoundaryAttribute::BOTTOM_3D)
        .export_values();

    // Structure CoarseFactors
    py::class_<CoarseFactors>(m, "CoarseFactors",
        "Facteurs d'homogénéisation pour chaque direction")
        .def(py::init<>())
        .def(py::init<int, int, int>(), py::arg("x"), py::arg("y"), py::arg("z"))
        .def_readwrite("x", &CoarseFactors::x)
        .def_readwrite("y", &CoarseFactors::y)
        .def_readwrite("z", &CoarseFactors::z);

    // ========================================================================
    // STRUCTURES POUR LES MODES PROPRES
    // ========================================================================

    py::class_<EigenMode>(m, "EigenMode",
        R"doc(
Résultat d'un mode propre (eigenvalue λ + eigenvector φ).

Le problème résolu est : L·φ = λ·F·φ

Attributes
----------
mode_index : int
    Index du mode (0 = fondamental, 1 = première harmonique, etc.)
lambda_ : float
    Eigenvalue λ_n. Pour le fondamental : λ₀ = 1/k_eff
keff : float
    Facteur de multiplication k = 1/λ
dominance_ratio : float
    Ratio λ_n/λ_0 (= k_0/k_n)
iterations : int
    Nombre d'itérations pour la convergence
converged : bool
    True si le mode a convergé
)doc")
        .def(py::init<>())
        .def_readonly("mode_index", &EigenMode::mode_index)
        .def_property_readonly("lambda_", [](const EigenMode& m) { return m.lambda; },
            "Eigenvalue λ (utiliser lambda_ car 'lambda' est réservé en Python)")
        .def_readonly("eigenvalue", &EigenMode::lambda,
            "Alias pour lambda_")
        .def_readonly("keff", &EigenMode::keff,
            "k-eff = 1/λ")
        .def_readonly("dominance_ratio", &EigenMode::dominance_ratio)
        .def_readonly("iterations", &EigenMode::iterations)
        .def_readonly("converged", &EigenMode::converged)
        .def("__repr__", [](const EigenMode& m) {
            return "<EigenMode n=" + std::to_string(m.mode_index) +
                   " λ=" + std::to_string(m.lambda) +
                   " k=" + std::to_string(m.keff) +
                   " converged=" + (m.converged ? "True" : "False") + ">";
        });

    py::class_<EigensolverParameters>(m, "EigensolverParameters",
        R"doc(
Paramètres pour le calcul des modes propres.

Parameters
----------
tol_eigenvalue : float, default=1e-5
    Tolérance relative sur λ : |Δλ|/|λ| < tol
tol_eigenvector : float, default=1e-4
    Tolérance sur les eigenvectors (non utilisé actuellement)
max_iter_per_mode : int, default=500
    Nombre maximum d'itérations par mode
deflation_weight : float, default=1.0
    Poids de déflation de Hotelling
orthogonalize : bool, default=True
    Orthogonaliser les modes avec produit <φ_i, F·φ_j>
normalize_eigenvectors : bool, default=True
    Normaliser les eigenvectors : ||φ|| = 1
)doc")
        .def(py::init<>())
        .def(py::init([](double tol_ev, double tol_vec, int max_iter,
                         double defl_weight, bool ortho, bool normalize) {
            EigensolverParameters p;
            p.tol_eigenvalue = tol_ev;
            p.tol_eigenvector = tol_vec;
            p.max_iter_per_mode = max_iter;
            p.deflation_weight = defl_weight;
            p.orthogonalize = ortho;
            p.normalize_eigenvectors = normalize;
            return p;
        }),
             py::arg("tol_eigenvalue") = 1e-5,
             py::arg("tol_eigenvector") = 1e-4,
             py::arg("max_iter_per_mode") = 500,
             py::arg("deflation_weight") = 1.0,
             py::arg("orthogonalize") = true,
             py::arg("normalize_eigenvectors") = true)
        .def_readwrite("tol_eigenvalue", &EigensolverParameters::tol_eigenvalue)
        .def_readwrite("tol_eigenvector", &EigensolverParameters::tol_eigenvector)
        .def_readwrite("max_iter_per_mode", &EigensolverParameters::max_iter_per_mode)
        .def_readwrite("deflation_weight", &EigensolverParameters::deflation_weight)
        .def_readwrite("orthogonalize", &EigensolverParameters::orthogonalize)
        .def_readwrite("normalize_eigenvectors", &EigensolverParameters::normalize_eigenvectors);

    // Classe principale NeutMFEM
    py::class_<NeutMFEM>(m, "NeutMFEM")
        // Constructeur
        .def(py::init<int, int, const std::vector<double>&,
                      const std::vector<double>&, const std::vector<double>&>(),
             py::arg("order"), py::arg("num_groups"),
             py::arg("x_breaks"), py::arg("y_breaks"), py::arg("z_breaks"),
             "Constructeur du solveur neutronique")

        // Configuration
        .def("set_bc", &NeutMFEM::SetBC,
             py::arg("attr"), py::arg("type"), py::arg("value") = 0.0)
        .def("set_robin_coefficients", &NeutMFEM::SetRobinCoefficients,
             py::arg("attr"), py::arg("alpha"), py::arg("beta"))
        .def("set_krylov_dimension", &NeutMFEM::SetKrylovDimension, py::arg("dim") = 5)
        .def("set_verbosity", &NeutMFEM::SetVerbosity, py::arg("level"))
        .def("set_tolerances", &NeutMFEM::SetTolerances,
             py::arg("tol_keff") = 1e-5, py::arg("tol_flux") = 1e-5,
             py::arg("tol_linear") = 1e-5, py::arg("max_outer") = 200,
             py::arg("max_inner") = 200)
        .def("reset_flux", &NeutMFEM::ResetFlux)

        // Symétries
        .def("apply_quarter_rotational_symmetry", &NeutMFEM::ApplyQuarterRotationalSymmetry,
             py::arg("axis1") = 0, py::arg("axis2") = 1)
        .def("apply_central_symmetry", &NeutMFEM::ApplyCentralSymmetry,
             py::arg("axis1") = 0, py::arg("axis2") = 1)

        // Réflecteurs
        .def("add_refl", &NeutMFEM::add_refl,
             py::arg("D"), py::arg("SigR"), py::arg("SigS"))
        .def("set_refl", &NeutMFEM::set_refl,
             py::arg("refl_id"), py::arg("dimension"), py::arg("is_upper"))
        .def("clean_refl", &NeutMFEM::clean_refl)

        // Construction et résolution
        .def("build_matrices", &NeutMFEM::BuildMatrices)

        .def("set_condensedform", &NeutMFEM::SetCondensedForm)

        // SolveKeff version simple
        .def("SolveKeff",
             static_cast<double (NeutMFEM::*)(LinearSolverType)>(&NeutMFEM::SolveKeff),
             py::arg("solver_type") = LinearSolverType::GMRES,
             "Résolution k-eff standard")

        // SolveKeff version complète avec CoarseFactors
        .def("SolveKeff",
             static_cast<double (NeutMFEM::*)(LinearSolverType, bool, const CoarseFactors&, 
                                             int, double)>(&NeutMFEM::SolveKeff),
             py::arg("solver_type") = LinearSolverType::GMRES,
             py::arg("use_coarse_init") = true,
             py::arg("coarse_factors") = CoarseFactors(17, 17, 17),
             py::arg("coarse_max_iter") = 30,
             py::arg("coarse_tol_keff") = 1e-3,
             "Résolution k-eff avec initialisation coarse (facteurs x/y/z)")

        // SolveKeff avec tuple Python pour les facteurs
        .def("SolveKeff",
             [](NeutMFEM& self, LinearSolverType solver_type, bool use_coarse,
                const py::object& factors, int max_iter, double tol) {
                 CoarseFactors cf = CoarseFactors::from_python(factors);
                 return self.SolveKeff(solver_type, use_coarse, cf, max_iter, tol);
             },
             py::arg("solver_type") = LinearSolverType::GMRES,
             py::arg("use_coarse_init") = true,
             py::arg("coarse_factors") = py::make_tuple(17, 17, 17),
             py::arg("coarse_max_iter") = 30,
             py::arg("coarse_tol_keff") = 1e-3,
             "Résolution k-eff avec facteurs coarse en tuple (fx, fy, fz)")

        .def("solve_adjoint", &NeutMFEM::SolveAdjoint,
             py::arg("solver_type") = LinearSolverType::GMRES,
             py::arg("normalize_to_direct") = true,
             py::arg("use_direct_keff") = true,
             "Résout le problème adjoint")

        .def("solve_FixedSource", &NeutMFEM::SolveFixedSource,
             py::arg("solver_type") = LinearSolverType::GMRES,
             "Résout avec sources fixées")

        .def("solve_subcritical", &NeutMFEM::SolveSubcritical,
             py::arg("solver_type") = LinearSolverType::GMRES,
             "Résout le problème sous-critique, retourne le facteur M")

        .def("zoom", &NeutMFEM::py_zoom,
             py::arg("refine"), py::arg("adjoint") = false,
             py::arg("solver_type") = LinearSolverType::GMRES)

// ==================================================================
        // CALCUL DES MODES PROPRES (EIGENVALUES λ ET EIGENVECTORS φ)
        // ==================================================================

        .def("solve_eigenmodes",
             static_cast<std::vector<EigenMode> (NeutMFEM::*)(int, LinearSolverType, 
                                                              const EigensolverParameters&)>
                         (&NeutMFEM::SolveEigenmodes),
             py::arg("num_modes"),
             py::arg("solver_type") = LinearSolverType::GMRES,
             py::arg("params") = EigensolverParameters(),
             R"doc(
Calcule les N premiers modes propres du problème L·φ = λ·F·φ.

Parameters
----------
num_modes : int
    Nombre de modes à calculer (incluant le fondamental)
solver_type : LinearSolverType, optional
    Type de solveur linéaire (défaut: GMRES)
params : EigensolverParameters, optional
    Paramètres du solveur

Returns
-------
list[EigenMode]
    Liste des modes propres avec eigenvalues λ et eigenvectors φ

Notes
-----
Les eigenvalues sont ordonnées : λ₀ < λ₁ < λ₂ < ...

Pour le mode fondamental: λ₀ = 1/k_eff
Le système est critique quand λ₀ = 1.

Examples
--------
>>> modes = solver.solve_eigenmodes(5)
>>> for m in modes:
...     print(f"λ_{m.mode_index} = {m.lambda_:.8f}")
...     print(f"k_{m.mode_index} = {m.keff:.8f}")
)doc")

        .def("solve_eigenmodes",
             static_cast<std::vector<EigenMode> (NeutMFEM::*)(int, LinearSolverType, bool,
                                                              const CoarseFactors&,
                                                              const EigensolverParameters&)>
                         (&NeutMFEM::SolveEigenmodes),
             py::arg("num_modes"),
             py::arg("solver_type") = LinearSolverType::GMRES,
             py::arg("use_coarse_init") = true,
             py::arg("coarse_factors") = CoarseFactors(17, 17, 17),
             py::arg("params") = EigensolverParameters(),
             "Calcule les modes propres avec initialisation coarse")

        .def("solve_eigenmodes",
             [](NeutMFEM& self, int num_modes, LinearSolverType solver_type,
                bool use_coarse, const py::object& factors, const EigensolverParameters& params) {
                 CoarseFactors cf = CoarseFactors::from_python(factors);
                 return self.SolveEigenmodes(num_modes, solver_type, use_coarse, cf, params);
             },
             py::arg("num_modes"),
             py::arg("solver_type") = LinearSolverType::GMRES,
             py::arg("use_coarse_init") = true,
             py::arg("coarse_factors") = py::make_tuple(17, 17, 17),
             py::arg("params") = EigensolverParameters(),
             "Calcule les modes propres avec facteurs coarse en tuple")

        .def("solve_adjoint_eigenmodes", &NeutMFEM::SolveAdjointEigenmodes,
             py::arg("num_modes"),
             py::arg("solver_type") = LinearSolverType::GMRES,
             py::arg("params") = EigensolverParameters(),
             "Calcule les modes propres adjoints L†·φ† = λ·F†·φ†")

        // ------------------------------------------------------------------
        // ACCESSEURS EIGENVALUES
        // ------------------------------------------------------------------

        .def("get_eigenvalues", &NeutMFEM::get_eigenvalues,
             R"doc(
Retourne toutes les eigenvalues λ calculées.

Returns
-------
list[float]
    [λ₀, λ₁, λ₂, ...] avec λ₀ < λ₁ < λ₂
)doc")

        .def("get_eigenvalue", &NeutMFEM::get_eigenvalue,
             py::arg("mode_index"),
             "Retourne l'eigenvalue λ_n du mode n")

        .def("get_keff_values", &NeutMFEM::get_keff_values,
             "Retourne les k-eff = 1/λ pour tous les modes")

        // ------------------------------------------------------------------
        // ACCESSEURS EIGENVECTORS
        // ------------------------------------------------------------------

        .def("get_eigenvector", &NeutMFEM::get_eigenvector,
             py::arg("mode_index"),
             R"doc(
Retourne l'eigenvector φ_n (flux du mode n).

Parameters
----------
mode_index : int
    Index du mode (0 = fondamental)

Returns
-------
numpy.ndarray
    Array shape (num_groups, [nz,] ny, nx)
)doc")

        .def("get_adjoint_eigenvector", &NeutMFEM::get_adjoint_eigenvector,
             py::arg("mode_index"),
             "Retourne l'eigenvector adjoint φ†_n")

        .def("get_num_eigenmodes", &NeutMFEM::get_num_eigenmodes,
             "Retourne le nombre de modes propres calculés")

        .def("get_dominance_ratio", &NeutMFEM::GetDominanceRatio,
             py::arg("mode_index"),
             R"doc(
Retourne le ratio de dominance λ_n/λ_0 (= k_0/k_n).

Un ratio proche de 1 indique une convergence lente.
)doc")

        .def("check_orthogonality", &NeutMFEM::CheckOrthogonality,
             R"doc(
Vérifie l'orthogonalité des modes (produit F-pondéré).

Returns
-------
list[list[float]]
    Matrice des produits scalaires <φ_i, F·φ_j>
    Devrait être diagonale si les modes sont orthogonaux.
)doc")

        .def("save_eigenvectors_vtk", &NeutMFEM::SaveEigenvectorsVTK,
             py::arg("filename_prefix"),
             py::arg("max_modes") = 0,
             "Exporte les eigenvectors en fichiers VTK")

        // Accesseurs (zero-copy)
        .def("get_D", &NeutMFEM::get_D)
        .def("get_src", &NeutMFEM::get_SRC)
        .def("get_SigR", &NeutMFEM::get_SigR)
        .def("get_NSF", &NeutMFEM::get_NSF)
        .def("get_KSF", &NeutMFEM::get_KSF)
        .def("get_Chi", &NeutMFEM::get_Chi)
        .def("get_SigS", &NeutMFEM::get_SigS)
        .def("get_flux", &NeutMFEM::get_flux)
        .def("get_flux_adj", &NeutMFEM::get_flux_adj)

        // Utilitaires
        .def("get_num_elements", &NeutMFEM::GetNumElements)
        .def("save_vtk", &NeutMFEM::SaveVTK, py::arg("filename_prefix"), py::arg("coarse") = true, py::arg("zoom") = true,
             "Export VTK du maillage principal (flux, XS)")
        .def("save_vtk_coarse", &NeutMFEM::SaveVTK_Coarse, py::arg("filename_prefix"),
             "Export VTK du maillage coarse (après homogénéisation)")
        .def("save_vtk_zoom", 
             static_cast<void (NeutMFEM::*)(std::string)>(&NeutMFEM::SaveVTK_Zoom),
             py::arg("filename_prefix"),
             "Export VTK du maillage zoomé")
        .def("get_group_importance", &NeutMFEM::GetGroupImportance, py::arg("group_idx"),
             "Calcule l'importance du groupe = <Φ†, M_fiss·Φ>");

    m.attr("__version__") = "0.0.1";
}
