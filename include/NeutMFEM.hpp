/**
 * @file NeutMFEM.hpp
 * @brief Solveur de diffusion neutronique multi-groupes par éléments finis mixtes
 * 
 * @author jujuc31
 * @version 0.0.1
 * @date 2026
 */

#ifndef NEUTMFEM_HPP
#define NEUTMFEM_HPP

#include "mfem.hpp"
#include <map>
#include <memory>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <string>
#include <vector>
#include <array>

namespace py = pybind11;

namespace neutmfem {

// ============================================================================
// ÉNUMÉRATIONS
// ============================================================================

enum class VerbosityLevel {
    SILENT = 0,
    LIGHT  = 1,
    NORMAL = 2,
    DEBUG  = 3
};

enum class LinearSolverType {
    BICGSTAB,
    GMRES,
    MINRES,
    CG,
    PCG,
    FGMRES
};

enum class BCType {
    DIRICHLET,
    NEUMANN,
    ROBIN,
    MIRROR
};

enum class BoundaryAttribute {
    LEFT_1D  = 1, RIGHT_1D = 2,
    LEFT_2D  = 1, RIGHT_2D = 2, TOP_2D    = 3, BOTTOM_2D = 4,
    FRONT_3D = 1, BACK_3D  = 2, LEFT_3D   = 3, RIGHT_3D  = 4, 
    TOP_3D   = 5, BOTTOM_3D = 6
};

// ============================================================================
// STRUCTURES AUXILIAIRES
// ============================================================================

/**
 * @struct CoarseFactors
 * @brief Facteurs d'homogénéisation pour chaque direction spatiale
 */
struct CoarseFactors {
    int x = 1;  ///< Facteur en X
    int y = 1;  ///< Facteur en Y
    int z = 1;  ///< Facteur en Z
    
    CoarseFactors() = default;
    CoarseFactors(int fx, int fy, int fz) : x(fx), y(fy), z(fz) {}
    
    /// Constructeur depuis tuple Python (fx, fy, fz) ou valeur unique
    static CoarseFactors from_python(const py::object& obj) {
        CoarseFactors f;
        if (py::isinstance<py::int_>(obj)) {
            int val = obj.cast<int>();
            f.x = f.y = f.z = val;
        } else if (py::isinstance<py::tuple>(obj) || py::isinstance<py::list>(obj)) {
            auto seq = obj.cast<std::vector<int>>();
            if (seq.size() >= 1) f.x = seq[0];
            if (seq.size() >= 2) f.y = seq[1];
            if (seq.size() >= 3) f.z = seq[2];
        }
        return f;
    }
};

/**
 * @struct ReflectorCoefficients
 * @brief Coefficients de conditions aux limites pour un réflecteur
 */
struct ReflectorCoefficients {
    mfem::Vector alpha;
    mfem::DenseMatrix beta;

    explicit ReflectorCoefficients(int n_grps_) 
        : alpha(n_grps_), beta(n_grps_, n_grps_) {
        alpha = 0.0;
        beta = 0.0;
    }
};

/**
 * @struct SolverParameters
 * @brief Paramètres de convergence pour les solveurs itératifs
 */
struct SolverParameters {
    double tol_keff       = 1e-5;
    double tol_flux       = 1e-5;
    double tol_linear     = 1e-5;
    int max_outer_iter    = 200;
    int max_inner_iter    = 200;
    int krylov_dimension  = 5;
};

/**
 * @struct EigenMode
 * @brief Résultat complet d'un mode propre (eigenvalue + eigenvector)
 */
struct EigenMode {
    int mode_index;              ///< Index du mode (0 = fondamental)
    double lambda;               ///< Eigenvalue λ_n (λ₀ = 1/k_eff)
    double keff;                 ///< k-eff associé = 1/λ
    double dominance_ratio;      ///< Ratio λ_n/λ_0 (ou k_0/k_n)
    int iterations;              ///< Nombre d'itérations pour converger
    bool converged;              ///< Indicateur de convergence
    
    EigenMode() : mode_index(0), lambda(1.0), keff(1.0), 
                  dominance_ratio(1.0), iterations(0), converged(false) {}
};

/**
 * @struct EigensolverParameters
 * @brief Paramètres pour le calcul des modes propres
 */
struct EigensolverParameters {
    double tol_eigenvalue = 1e-5;     ///< Tolérance sur λ
    double tol_eigenvector = 1e-4;    ///< Tolérance sur φ (norme L2)
    int max_iter_per_mode = 500;      ///< Itérations max par mode
    double deflation_weight = 1.0;    ///< Poids de déflation
    bool orthogonalize = true;        ///< Gram-Schmidt sur les modes
    bool normalize_eigenvectors = true; ///< Normaliser ||φ||=1
};

// ============================================================================
// CLASSE PRINCIPALE
// ============================================================================

class NeutMFEM {
public:
    // ========================================================================
    // CONSTRUCTEUR ET DESTRUCTEUR
    // ========================================================================

    NeutMFEM(int order, int n_grps_,
            const std::vector<double>& x_breaks,
            const std::vector<double>& y_breaks,
            const std::vector<double>& z_breaks);

    ~NeutMFEM();

    NeutMFEM(const NeutMFEM&) = delete;
    NeutMFEM& operator=(const NeutMFEM&) = delete;

    // ========================================================================
    // CONFIGURATION
    // ========================================================================

    void SetBC(int attr, BCType type, double value = 0.0);
    void SetRobinCoefficients(int attr, double alpha, double beta);
    void SetKrylovDimension(int dim);
    void SetVerbosity(VerbosityLevel level) { verbosity_ = level; }
    
    void SetTolerances(double tol_keff = 1e-5, double tol_flux = 1e-5,
                       double tol_linear = 1e-5, 
                       int max_outer = 200, int max_inner = 200);

    /**
     * @brief Active ou désactive la forme condensée
     * 
     * Forme condensée: K*Phi = S où K = B*A^{-1}*B^T + C
     * Plus efficace que le système mixte flux/courant pour la plupart des cas.
     * 
     * @param use_condensed true pour utiliser la forme condensée (défaut)
     */
    void SetCondensedForm(bool use_condensed) { use_condensed_form_ = use_condensed; }
    
    /// Retourne true si la forme condensée est activée
    [[nodiscard]] bool IsCondensedForm() const { return use_condensed_form_; }

    void ResetFlux();

    // ========================================================================
    // SYMÉTRIES
    // ========================================================================

    void ApplyQuarterRotationalSymmetry(int axis1 = 0, int axis2 = 1);
    void ApplyCentralSymmetry(int axis1 = 0, int axis2 = 1);

    // ========================================================================
    // RÉFLECTEURS
    // ========================================================================

    int add_refl(py::array_t<double> D, py::array_t<double> SigR,
                 py::array_t<double> SigS);
    void set_refl(int refl_id, int dimension, bool is_upper);
    void clean_refl();

    // ========================================================================
    // CONSTRUCTION ET RÉSOLUTION
    // ========================================================================

    void BuildMatrices();

    /// Version simple
    double SolveKeff(LinearSolverType solver_type);

    /**
     * @brief Résolution k-eff avec initialisation coarse
     * 
     * @param solver_type Type de solveur linéaire
     * @param use_coarse_init Activer l'initialisation grossière
     * @param coarse_factors Facteurs d'homogénéisation (x, y, z)
     * @param coarse_max_iter Itérations max pour la phase grossière
     * @param coarse_tol_keff Tolérance k-eff pour la phase grossière
     */
    double SolveKeff(LinearSolverType solver_type,
                     bool use_coarse_init,
                     const CoarseFactors& coarse_factors,
                     int coarse_max_iter,
                     double coarse_tol_keff);

    /**
     * @brief Résout le problème adjoint
     * 
     * @param solver_type Type de solveur linéaire
     * @param normalize_to_direct Normaliser par rapport au flux direct
     * @param use_direct_keff Utiliser k-eff du problème direct si disponible
     * @return k-eff adjoint
     */
    double SolveAdjoint(LinearSolverType solver_type = LinearSolverType::GMRES,
                        bool normalize_to_direct = true,
                        bool use_direct_keff = true);

    /**
     * @brief Résout avec sources externes fixées (pas d'itérations k-eff)
     * @param solver_type Type de solveur linéaire
     */
    void SolveFixedSource(LinearSolverType solver_type = LinearSolverType::GMRES);

    /**
     * @brief Reconstruit les courants J à partir des flux Phi
     * 
     * Utile après une résolution en forme condensée si les courants sont nécessaires.
     * Résout A * J = B^T * Phi pour chaque groupe.
     */
    void ReconstructAllCurrents();

    /**
     * @brief Résout le problème sous-critique
     * 
     * Calcule le facteur de multiplication M = Production / Source_externe
     * 
     * @param solver_type Type de solveur linéaire
     * @return Facteur de multiplication sous-critique M
     */
    double SolveSubcritical(LinearSolverType solver_type = LinearSolverType::GMRES);

    /**
     * @brief Raffinement du maillage avec sources figées (zoom)
     */
    mfem::GridFunction* zoom(std::vector<int> refine_factors,
                             LinearSolverType solver_type = LinearSolverType::GMRES);

    py::array_t<double> py_zoom(py::tuple refine, bool adjoint = false,
                                LinearSolverType solver_type = LinearSolverType::GMRES);

    // ========================================================================
    // ACCESSEURS PYTHON (ZERO-COPY)
    // ========================================================================

    py::array_t<double> get_D();
    py::array_t<double> get_SRC();
    py::array_t<double> get_SigR();
    py::array_t<double> get_NSF();
    py::array_t<double> get_KSF();
    py::array_t<double> get_Chi();
    py::array_t<double> get_SigS();
    py::array_t<double> get_flux();
    py::array_t<double> get_flux_adj();

    // ========================================================================
    // UTILITAIRES
    // ========================================================================

    int GetNumElements() const;
    
    /// Export VTK du maillage principal (flux, XS)
    void SaveVTK(std::string filename_prefix, bool coarse = true, bool zoom = true);
    
    /// Export VTK du maillage coarse (après homogénéisation)
    void SaveVTK_Coarse(std::string filename_prefix);
    
    /// Export VTK du maillage zoomé (après zoom)
    void SaveVTK_Zoom(std::string filename_prefix);
    
    /// Export VTK du maillage zoomé avec données externes
    void SaveVTK_Zoom(std::string filename_prefix, 
                      mfem::GridFunction* phi_zoom,
                      const std::vector<int>& refine_factors);
    
    /**
     * @brief Calcule l'importance relative d'un groupe
     * @param group_idx Index du groupe (0-based)
     * @return Coefficient d'importance = <Φ†, M_fiss·Φ>
     */
    double GetGroupImportance(int group_idx);

    /**
    	 * @brief Calcule les N premiers modes propres du problème L·φ = λ·F·φ
    	 * 
    	 * @param num_modes Nombre de modes à calculer
    	 * @param solver_type Type de solveur linéaire
    	 * @param params Paramètres du solveur
    	 * @return Vecteur des modes propres (eigenvalue λ + eigenvector φ)
    	 * 
    	 * @note Les eigenvalues sont ordonnées : λ₀ < λ₁ < λ₂ < ...
    	 *	 Le mode fondamental a λ₀ = 1/k_eff
    	 */
    	std::vector<EigenMode> SolveEigenmodes(
    	    int num_modes,
    	    LinearSolverType solver_type = LinearSolverType::GMRES,
    	    const EigensolverParameters& params = EigensolverParameters());
    
    	/**
    	 * @brief Version avec initialisation coarse
    	 */
    	std::vector<EigenMode> SolveEigenmodes(
    	    int num_modes,
    	    LinearSolverType solver_type,
    	    bool use_coarse_init,
    	    const CoarseFactors& coarse_factors,
    	    const EigensolverParameters& params = EigensolverParameters());
    
    	/**
    	 * @brief Calcule les modes propres adjoints : L†·φ† = λ·F†·φ†
    	 */
    	std::vector<EigenMode> SolveAdjointEigenmodes(
    	    int num_modes,
    	    LinearSolverType solver_type = LinearSolverType::GMRES,
    	    const EigensolverParameters& params = EigensolverParameters());
    
    	// ------------------------------------------------------------------------
    	// Accesseurs pour les eigenvalues (λ)
    	// ------------------------------------------------------------------------
    
    	/**
    	 * @brief Retourne toutes les eigenvalues λ calculées
    	 * @return Vecteur {λ₀, λ₁, λ₂, ...} avec λ₀ < λ₁ < λ₂
    	 */
    	std::vector<double> get_eigenvalues() const { return eigenvalues_; }
    
    	/**
    	 * @brief Retourne l'eigenvalue λ d'un mode spécifique
    	 */
    	double get_eigenvalue(int mode_index) const;
    
    	/**
    	 * @brief Retourne les eigenvalues converties en k-eff (= 1/λ)
    	 */
    	std::vector<double> get_keff_values() const;
    
    	// ------------------------------------------------------------------------
    	// Accesseurs pour les eigenvectors (φ)
    	// ------------------------------------------------------------------------
    
    	/**
    	 * @brief Retourne l'eigenvector φ_n (flux du mode n)
    	 * @param mode_index Index du mode (0 = fondamental)
    	 * @return Array numpy shape (num_groups, [nz,] ny, nx)
    	 */
    	py::array_t<double> get_eigenvector(int mode_index);
    
    	/**
    	 * @brief Retourne l'eigenvector adjoint φ†_n
    	 */
    	py::array_t<double> get_adjoint_eigenvector(int mode_index);

    	/**
    	 * @brief Nombre de modes propres calculés
    	 */
    	int get_num_eigenmodes() const { return static_cast<int>(eigenvalues_.size()); }
    
    	// ------------------------------------------------------------------------
    	// Utilitaires
    	// ------------------------------------------------------------------------
    
    	/**
    	 * @brief Ratio de dominance λ_n / λ_0 (équivalent à k_0 / k_n)
    	 */
    	double GetDominanceRatio(int mode_index) const;
    
    	/**
    	 * @brief Vérifie l'orthogonalité des modes (produit F-pondéré)
    	 * @return Matrice des produits scalaires <φ_i, F·φ_j>
    	 */
    	std::vector<std::vector<double>> CheckOrthogonality() const;
    
    	/**
    	 * @brief Export VTK de tous les eigenvectors
    	 */
    	void SaveEigenvectorsVTK(const std::string& filename_prefix, int max_modes = 0);
 
private:
    // ========================================================================
    // DONNÉES MEMBRES - MAILLAGE ET GÉOMÉTRIE
    // ========================================================================
    
    mfem::Mesh* mesh_;
    int nx_, ny_, nz_;
    int n_grps_;

    // ========================================================================
    // DONNÉES MEMBRES - ÉLÉMENTS FINIS
    // ========================================================================
    
    mfem::RT_FECollection* fec_J_;
    mfem::L2_FECollection* fec_Phi_;
    mfem::L2_FECollection* fec_Mat_;

    mfem::FiniteElementSpace* fes_J_;
    mfem::FiniteElementSpace* fes_Phi_;
    mfem::FiniteElementSpace* fes_Mat_;

    mfem::FiniteElementSpace* fes_J_mg_;
    mfem::FiniteElementSpace* fes_Phi_mg_;
    mfem::FiniteElementSpace* fes_Mat_mg_;
    mfem::FiniteElementSpace* fes_SigS_;

    // ========================================================================
    // DONNÉES MEMBRES - GRIDFUNCTIONS
    // ========================================================================
    
    mfem::GridFunction* D_gf_;
    mfem::GridFunction* SRC_gf_;
    mfem::GridFunction* SigR_gf_;
    mfem::GridFunction* NSF_gf_;
    mfem::GridFunction* KSF_gf_;
    mfem::GridFunction* Chi_gf_;
    mfem::GridFunction* SigS_gf_;

    mfem::GridFunction* Sol_Phi_gf_;
    mfem::GridFunction* Sol_J_gf_;
    mfem::GridFunction* Sol_Phi_adj_gf_;
    mfem::GridFunction* Sol_J_adj_gf_;

    mfem::GridFunction* Sol_Phi_coarse_gf_;
    mfem::GridFunction* D_coarse_gf_;
    mfem::GridFunction* NSF_coarse_gf_;
    mfem::GridFunction* Chi_coarse_gf_;
    mfem::GridFunction* SigR_coarse_gf_;
    mfem::GridFunction* SigS_coarse_gf_;
    mfem::GridFunction* SRC_coarse_gf_;
    CoarseFactors coarse_factors_used_;
    
    // Données du maillage coarse (pour export VTK)
    mfem::Mesh* mesh_coarse_;
    mfem::FiniteElementSpace* fes_Mat_coarse_;
    mfem::FiniteElementSpace* fes_Phi_coarse_;
    int nx_coarse_, ny_coarse_, nz_coarse_;
    
    // Données du maillage zoom (pour export VTK)
    mfem::Mesh* mesh_zoom_;
    mfem::FiniteElementSpace* fes_Phi_zoom_;
    mfem::GridFunction* Sol_Phi_zoom_gf_;
    std::vector<int> zoom_factors_;

    mfem::GridFunctionCoefficient* D_work_coeff_;
    mfem::GridFunctionCoefficient* SigR_work_coeff_;
    mfem::GridFunction* D_work_gf_;
    mfem::GridFunction* SigR_work_gf_;

    // ========================================================================
    // DONNÉES MEMBRES - MATRICES
    // ========================================================================
    
    mfem::SparseMatrix* B_mat_;
    mfem::SparseMatrix* BT_mat_;
    std::vector<mfem::SparseMatrix*> A_mats_;
    std::vector<mfem::SparseMatrix*> C_mats_;
    std::vector<mfem::SparseMatrix*> M_fiss_;
    std::vector<mfem::SparseMatrix*> M_scatter_;
    
    // Forme condensée : K_g = B * A_g^{-1} * B^T + C_g
    std::vector<mfem::SparseMatrix*> K_condensed_;  ///< Matrices condensées par groupe
    bool use_condensed_form_;  ///< Active la forme condensée (défaut: true)

    // ========================================================================
    // DONNÉES MEMBRES - CONDITIONS AUX LIMITES
    // ========================================================================
    
    mfem::Array<int> ess_bdr_J_;
    mfem::Array<int> ess_bdr_Phi_;
    mfem::Array<int> nat_bdr_J_;
    mfem::Array<int> ess_tdof_list_J_;
    mfem::Array<int> ess_tdof_list_Phi_;
    mfem::Vector dirichlet_phi_vals_;

    std::map<int, BCType> boundary_types_;
    std::map<int, double> boundary_values_;
    std::map<int, double> dirichlet_values_;
    std::map<int, double> neumann_values_;
    std::map<int, double> robin_alpha_;
    std::map<int, double> robin_beta_;
    std::map<int, double> robin_values_;

    std::vector<mfem::LinearForm*> neumann_lf_J_;
    std::vector<mfem::LinearForm*> robin_lf_Phi_;

    // ========================================================================
    // DONNÉES MEMBRES - RÉFLECTEURS
    // ========================================================================
    
    std::vector<ReflectorCoefficients> reflector_types_;
    std::map<std::pair<int,int>, int> active_reflectors_;
    std::map<std::pair<int,int>, mfem::Vector> reflector_rhs_;

    // ========================================================================
    // DONNÉES MEMBRES - PARAMÈTRES
    // ========================================================================
    
    SolverParameters params_;
    VerbosityLevel verbosity_;
    
    bool has_quarter_symmetry_;
    bool has_central_symmetry_;
    int sym_axis1_, sym_axis2_;
    int central_axis1_, central_axis2_;

    double last_keff_direct_;
    bool has_valid_keff_;

    /// Eigenvectors φ_n pour chaque mode
    std::vector<mfem::GridFunction*> eigenvectors_;
    
    /// Eigenvectors adjoints φ†_n
    std::vector<mfem::GridFunction*> eigenvectors_adj_;
    
    /// Eigenvalues λ_n (λ₀ = 1/k_eff)
    std::vector<double> eigenvalues_;
    
    /// Eigenvalues adjointes
    std::vector<double> eigenvalues_adj_;
    
    /// Paramètres du eigensolver
    EigensolverParameters eigensolver_params_;

    // ========================================================================
    // MÉTHODES PRIVÉES - UTILITAIRES
    // ========================================================================
    
    [[nodiscard]] int GetGroupOffset(int g) const { 
        return g * mesh_->GetNE(); 
    }
    
    [[nodiscard]] int GetGroupOffsetPhi(int g) const { 
        return g * fes_Phi_->GetVSize(); 
    }
    
    [[nodiscard]] int GetGroupOffsetJ(int g) const { 
        return g * fes_J_->GetVSize(); 
    }
    
    [[nodiscard]] int GetSigSOffset(int g_from, int g_to) const {
        return (g_to * n_grps_ + g_from) * mesh_->GetNE();
    }

    [[nodiscard]] int GetBoundaryAttribute(int mesh_dim, int dimension, 
                                           bool is_upper) const;

    template<typename... Args>
    void Log(VerbosityLevel level, Args&&... args) const {
        if (static_cast<int>(verbosity_) >= static_cast<int>(level)) {
            (std::cout << ... << std::forward<Args>(args)) << std::endl;
        }
    }

    // ========================================================================
    // MÉTHODES PRIVÉES - RÉSOLUTION
    // ========================================================================
    
    mfem::IterativeSolver* CreateLinearSolver(LinearSolverType type);
    
    void SolveGroupInternal(int g_idx, const mfem::Vector& integrated_src,
                            LinearSolverType solver_type);
    
    /// Résolution forme condensée : K_g * Phi_g = RHS
    void SolveGroupCondensed(int g_idx, const mfem::Vector& integrated_src,
                             LinearSolverType solver_type);
    
    /// Reconstruit le courant J à partir de Phi (post-traitement)
    void ReconstructCurrent(int g_idx);
    
    void SolveAdjointGroupInternal(int g_idx, const mfem::Vector& integrated_src,
                                   LinearSolverType solver_type);
    
    /// Résolution adjointe forme condensée
    void SolveAdjointGroupCondensed(int g_idx, const mfem::Vector& integrated_src,
                                    LinearSolverType solver_type);
    
    void SolveAdjointFixed(double keff_known, LinearSolverType solver_type,
                           bool normalize_to_direct);
    
    double ComputeAdjointProduction();

    // ========================================================================
    // MÉTHODES PRIVÉES - INTERPOLATION
    // ========================================================================
    
    py::array_t<double> GetCoefArray(mfem::GridFunction* gf,
                                     const std::vector<int>& subdiv = {1,1,1});
    
    void InterpolateConstantToRefinedMesh(const mfem::GridFunction& coarse_gf,
                                          mfem::GridFunction& fine_gf,
                                          const std::vector<int>& factors) const;
    
    /**
     * @brief Homogénéise vers un maillage grossier avec facteurs x/y/z séparés
     * 
     * @param fine_gf GridFunction sur maillage fin
     * @param factors Facteurs d'homogénéisation (x, y, z)
     * @param coarse_values Vecteur de sortie
     * @param volume_weighted Pondération par volume (défaut: true)
     */
    void HomogenizeToCoarse(const mfem::GridFunction& fine_gf,
                            const CoarseFactors& factors,
                            std::vector<double>& coarse_values,
                            bool volume_weighted = true) const;
    
    /**
     * @brief Interpole depuis un maillage grossier avec facteurs x/y/z séparés
     */
    void InterpolateFromCoarse(const std::vector<double>& coarse_values,
                               const CoarseFactors& factors,
                               mfem::GridFunction& fine_gf) const;
    
    /**
     * @brief Calcule les breaks pour un maillage grossier
     */
    std::vector<double> ComputeCoarseBreaks(const std::vector<double>& fine_breaks,
                                            int coarse_factor) const;

    // ========================================================================
    // MÉTHODES PRIVÉES - RÉFLECTEURS
    // ========================================================================
    
    void update_refl(const mfem::GridFunction& phi_current, int group_idx);

    void ExtractGroupFromMultiGroup(mfem::GridFunction* multi_gf, int group_idx,
                                    mfem::GridFunction* single_gf);
    void InjectGroupIntoMultiGroup(mfem::GridFunction* single_gf, int group_idx,
                                   mfem::GridFunction* multi_gf);

    	/**
    	 * @brief Calcule un mode propre avec déflation
    	 * 
    	 * Algorithme : Itération de puissance inverse sur (L - σF)⁻¹F
    	 * avec déflation de Hotelling pour les modes supérieurs.
    	 */
    	EigenMode ComputeSingleEigenmode(
    	    int mode_index,
    	    LinearSolverType solver_type,
    	    const std::vector<mfem::GridFunction*>& previous_modes,
    	    const std::vector<double>& previous_eigenvalues);
    
    	/**
    	 * @brief Applique la déflation de Hotelling
    	 */
    	void ApplyHotellingDeflation(
    	    mfem::Vector& source,
    	    const std::vector<mfem::GridFunction*>& previous_modes,
    	    const std::vector<double>& previous_eigenvalues);
    
    	/**
    	 * @brief Orthogonalise φ par rapport aux modes précédents
    	 * 
    	 * Utilise le produit scalaire F-pondéré : <φ, ψ>_F = <φ, F·ψ>
    	 */
    	void OrthogonalizeMode(
    	    mfem::GridFunction& phi,
    	    const std::vector<mfem::GridFunction*>& previous_modes);

    	/**
    	 * @brief Produit scalaire pondéré par F : <φ₁, F·φ₂>
    	 */
    	double FissionInnerProduct(
    	    const mfem::GridFunction& phi1,
    	    const mfem::GridFunction& phi2);
    
    	/**
    	 * @brief Nettoie les données des modes propres
    	 */
    	void ClearEigenmodeData();

    	/**
    	 * @brief Calcule un mode adjoint avec déflation
    	 */
    	EigenMode ComputeSingleAdjointEigenmode(
    	    int mode_index,
    	    LinearSolverType solver_type,
    	    const std::vector<mfem::GridFunction*>& previous_modes,
    	    const std::vector<double>& previous_eigenvalues);
};

} // namespace neutmfem

// Alias pour compatibilité
using VerbosityLevel = neutmfem::VerbosityLevel;
using LinearSolverType = neutmfem::LinearSolverType;
using BCType = neutmfem::BCType;
using BoundaryAttribute = neutmfem::BoundaryAttribute;
using CoarseFactors = neutmfem::CoarseFactors;

#endif // NEUTMFEM_HPP
