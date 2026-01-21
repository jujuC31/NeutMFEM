/**
 * @file NeutMFEM.cpp
 * @brief Implémentation du solveur de diffusion neutronique multi-groupes
 * 
 * @author jujuC31
 * @version 0.0.1
 * @date 2026
 */

#include "NeutMFEM.hpp"
#include "solver.hpp"
#include "mfem.hpp"
#include <cmath>
#include <cstring>
#include <iostream>
#include <iomanip>
#include <stdexcept>
#include <set>
#include <algorithm>

namespace neutmfem{

// ============================================================================
// CLASSE HELPER : COEFFICIENT INVERSE
// ============================================================================

class InverseCoefficient : public mfem::Coefficient {
public:
    explicit InverseCoefficient(mfem::GridFunction* gf) : gf_(gf) {}

    double Eval(mfem::ElementTransformation& T,
                const mfem::IntegrationPoint& ip) override {
        const double val = gf_->GetValue(T, ip);
        return (std::abs(val) > 1e-12) ? 1.0 / val : 0.0;
    }

private:
    mfem::GridFunction* gf_;
};

// ============================================================================
// CLASSE HELPER : OPÉRATEUR CONDENSÉ K = B * A^{-1} * B^T + C
// Applique l'opérateur sans jamais former la matrice explicitement
// ============================================================================

class CondensedOperator : public mfem::Operator {
public:
    CondensedOperator(mfem::SparseMatrix* A, mfem::SparseMatrix* B, 
                      mfem::SparseMatrix* BT, mfem::SparseMatrix* C,
                      double tol = 1e-5, int max_iter = 200)
        : mfem::Operator(C->Height(), C->Width()),
          A_(A), B_(B), BT_(BT), C_(C),
          tol_(tol), max_iter_(max_iter),
          tmp_J_(A->Height()), tmp_J2_(A->Height()) 
    {
        // Préconditionneur pour A^{-1}
        //A_prec_ = new mfem::DSmoother(*A_, 1);
        A_solver_ = new mfem::CGSolver();
        A_solver_->SetOperator(*A_);
        //A_solver_->SetPreconditioner(*A_prec_);
        A_solver_->SetRelTol(tol_);
        A_solver_->SetMaxIter(max_iter_);
        A_solver_->SetPrintLevel(-1);
    }
    
    ~CondensedOperator() {
        delete A_solver_;
        //delete A_prec_;
    }
    
    // Applique K * x = (B * A^{-1} * B^T + C) * x
    void Mult(const mfem::Vector& x, mfem::Vector& y) const override {
        // y = C * x
        C_->Mult(x, y);
        
        // tmp_J = B^T * x
        BT_->Mult(x, tmp_J_);
        
        // tmp_J2 = A^{-1} * tmp_J  (résout A * tmp_J2 = tmp_J)
        tmp_J2_ = 0.0;
        A_solver_->Mult(tmp_J_, tmp_J2_);
        
        // y += B * tmp_J2
        B_->AddMult(tmp_J2_, y);
    }

private:
    mfem::SparseMatrix* A_;
    mfem::SparseMatrix* B_;
    mfem::SparseMatrix* BT_;
    mfem::SparseMatrix* C_;
    double tol_;
    int max_iter_;
    
    mfem::DSmoother* A_prec_;
    mfem::CGSolver* A_solver_;
    
    mutable mfem::Vector tmp_J_;
    mutable mfem::Vector tmp_J2_;
};

// ============================================================================
// CONSTRUCTEUR
// ============================================================================

NeutMFEM::NeutMFEM(int order, int n_grps,
                 const std::vector<double>& x_breaks,
                 const std::vector<double>& y_breaks,
                 const std::vector<double>& z_breaks)
    : n_grps_(n_grps)
    , verbosity_(VerbosityLevel::NORMAL)
    , has_quarter_symmetry_(false)
    , has_central_symmetry_(false)
    , sym_axis1_(0), sym_axis2_(1)
    , central_axis1_(0), central_axis2_(1)
    , last_keff_direct_(-1.0)
    , has_valid_keff_(false)
    , use_condensed_form_(true) {  // Forme condensée par défaut

    if (n_grps <= 0) {
        throw std::invalid_argument("Le nombre de groupes doit être positif");
    }
    if (x_breaks.size() < 2) {
        throw std::invalid_argument("x_breaks doit contenir au moins 2 valeurs");
    }

    // Détection de la dimension
    nx_ = static_cast<int>(x_breaks.size()) - 1;
    ny_ = static_cast<int>(y_breaks.size()) - 1;
    nz_ = static_cast<int>(z_breaks.size()) - 1;
    
    int dim = 3;
    if (nz_ <= 0) { nz_ = 1; dim = 2; }
    if (ny_ <= 0) { ny_ = 1; dim = 1; }

    Log(VerbosityLevel::NORMAL, "=== INITIALISATION DU SOLVEUR ===");
    Log(VerbosityLevel::NORMAL, "Dimension : ", dim, "D");
    Log(VerbosityLevel::NORMAL, "Maillage : ", nx_, " × ", ny_, " × ", nz_);
    Log(VerbosityLevel::NORMAL, "Groupes d'énergie : ", n_grps);

    // Création du maillage
    if (dim == 3) {
        mesh_ = new mfem::Mesh(mfem::Mesh::MakeCartesian3D(
            nx_, ny_, nz_, mfem::Element::HEXAHEDRON, 1.0, 1.0, 1.0, false));
    } else if (dim == 2) {
        mesh_ = new mfem::Mesh(mfem::Mesh::MakeCartesian2D(
            nx_, ny_, mfem::Element::QUADRILATERAL, false, 1.0, 1.0, false));
    } else {
        mesh_ = new mfem::Mesh(mfem::Mesh::MakeCartesian1D(nx_, 1.0));
    }

    // Mise à jour des coordonnées
    for (int i = 0; i < mesh_->GetNV(); ++i) {
        double* v = mesh_->GetVertex(i);
        int ix = std::min(static_cast<int>(std::round(v[0] * nx_)), nx_);
        v[0] = x_breaks[ix];
        if (dim >= 2) {
            int iy = std::min(static_cast<int>(std::round(v[1] * ny_)), ny_);
            v[1] = y_breaks[iy];
        }
        if (dim == 3) {
            int iz = std::min(static_cast<int>(std::round(v[2] * nz_)), nz_);
            v[2] = z_breaks[iz];
        }
    }
    mesh_->Finalize(true);

    // Espaces éléments finis
    fec_J_   = new mfem::RT_FECollection(order, dim);
    fec_Phi_ = new mfem::L2_FECollection(order, dim);
    fec_Mat_ = new mfem::L2_FECollection(0, dim);

    fes_J_   = new mfem::FiniteElementSpace(mesh_, fec_J_);
    fes_Phi_ = new mfem::FiniteElementSpace(mesh_, fec_Phi_);
    fes_Mat_ = new mfem::FiniteElementSpace(mesh_, fec_Mat_);

    fes_J_mg_ = new mfem::FiniteElementSpace(
        mesh_, fec_J_, n_grps, mfem::Ordering::byNODES);
    fes_Phi_mg_ = new mfem::FiniteElementSpace(
        mesh_, fec_Phi_, n_grps, mfem::Ordering::byNODES);
    fes_Mat_mg_ = new mfem::FiniteElementSpace(
        mesh_, fec_Mat_, n_grps, mfem::Ordering::byNODES);
    fes_SigS_ = new mfem::FiniteElementSpace(
        mesh_, fec_Mat_, n_grps * n_grps, mfem::Ordering::byNODES);

    // GridFunctions
    D_gf_    = new mfem::GridFunction(fes_Mat_mg_);
    SRC_gf_  = new mfem::GridFunction(fes_Mat_mg_);
    SigR_gf_ = new mfem::GridFunction(fes_Mat_mg_);
    NSF_gf_  = new mfem::GridFunction(fes_Mat_mg_);
    KSF_gf_  = new mfem::GridFunction(fes_Mat_mg_);
    Chi_gf_  = new mfem::GridFunction(fes_Mat_mg_);
    SigS_gf_ = new mfem::GridFunction(fes_SigS_);

    Sol_Phi_gf_     = new mfem::GridFunction(fes_Phi_mg_);
    Sol_J_gf_       = new mfem::GridFunction(fes_J_mg_);
    Sol_Phi_adj_gf_ = new mfem::GridFunction(fes_Phi_mg_);
    Sol_J_adj_gf_   = new mfem::GridFunction(fes_J_mg_);

    Sol_Phi_coarse_gf_ = nullptr;
    D_coarse_gf_       = nullptr;
    NSF_coarse_gf_     = nullptr;
    Chi_coarse_gf_     = nullptr;
    SigR_coarse_gf_    = nullptr;
    SigS_coarse_gf_    = nullptr;
    SRC_coarse_gf_     = nullptr;
    
    mesh_coarse_       = nullptr;
    fes_Mat_coarse_    = nullptr;
    fes_Phi_coarse_    = nullptr;
    nx_coarse_ = ny_coarse_ = nz_coarse_ = 0;
    
    mesh_zoom_         = nullptr;
    fes_Phi_zoom_      = nullptr;
    Sol_Phi_zoom_gf_   = nullptr;

    D_work_coeff_    = nullptr;
    SigR_work_coeff_ = nullptr;
    D_work_gf_       = new mfem::GridFunction(fes_Phi_);
    SigR_work_gf_    = new mfem::GridFunction(fes_Phi_);

    // Initialisation
    *D_gf_    = 1.0;
    *SRC_gf_  = 0.0;
    *SigR_gf_ = 0.01;
    *NSF_gf_  = 0.0;
    *KSF_gf_  = 0.0;
    *Chi_gf_  = 0.0;
    *SigS_gf_ = 0.0;
    *Sol_Phi_gf_     = 1.0;
    *Sol_J_gf_       = 0.0;
    *Sol_Phi_adj_gf_ = 1.0;
    *Sol_J_adj_gf_   = 0.0;

    B_mat_  = nullptr;
    BT_mat_ = nullptr;
    A_mats_.assign(n_grps, nullptr);
    C_mats_.assign(n_grps, nullptr);
    K_condensed_.assign(n_grps, nullptr);  // Matrices condensées
    M_fiss_.assign(n_grps, nullptr);
    M_scatter_.assign(n_grps * n_grps, nullptr);

    ess_bdr_J_.SetSize(0);
    ess_bdr_Phi_.SetSize(0);
    nat_bdr_J_.SetSize(0);

    Log(VerbosityLevel::NORMAL, "Initialisation terminée avec succès\n");
}

// ============================================================================
// DESTRUCTEUR
// ============================================================================

NeutMFEM::~NeutMFEM() {
    delete D_gf_; delete SRC_gf_; delete SigR_gf_;
    delete NSF_gf_; delete KSF_gf_; delete Chi_gf_; delete SigS_gf_;
    delete Sol_Phi_gf_; delete Sol_J_gf_;
    delete Sol_Phi_adj_gf_; delete Sol_J_adj_gf_;
    delete Sol_Phi_coarse_gf_; delete D_coarse_gf_;
    delete NSF_coarse_gf_; delete Chi_coarse_gf_; delete SigR_coarse_gf_;
    delete SigS_coarse_gf_; delete SRC_coarse_gf_;
    delete D_work_coeff_; delete SigR_work_coeff_;
    delete D_work_gf_; delete SigR_work_gf_;
    
    // Nettoyage coarse
    delete fes_Mat_coarse_; delete fes_Phi_coarse_;
    delete mesh_coarse_;
    
    // Nettoyage zoom
    delete Sol_Phi_zoom_gf_;
    delete fes_Phi_zoom_;
    delete mesh_zoom_;

    for (auto& mat : A_mats_) delete mat;
    for (auto& mat : C_mats_) delete mat;
    for (auto& mat : K_condensed_) delete mat;  // Nettoyage matrices condensées
    for (auto& mat : M_fiss_) delete mat;
    for (auto& mat : M_scatter_) delete mat;
    for (auto& lf : neumann_lf_J_) delete lf;
    for (auto& lf : robin_lf_Phi_) delete lf;

    delete B_mat_; delete BT_mat_;
    delete fes_J_; delete fes_Phi_; delete fes_Mat_;
    delete fes_J_mg_; delete fes_Phi_mg_; delete fes_Mat_mg_; delete fes_SigS_;
    delete fec_J_; delete fec_Phi_; delete fec_Mat_;
    delete mesh_;
}

// ============================================================================
// CONFIGURATION
// ============================================================================

void NeutMFEM::ResetFlux() {
    *Sol_Phi_gf_ = 1.0; *Sol_J_gf_ = 0.0;
    *Sol_Phi_adj_gf_ = 1.0; *Sol_J_adj_gf_ = 0.0;
    last_keff_direct_ = 1.0; has_valid_keff_ = false;
}

void NeutMFEM::SetTolerances(double tol_keff, double tol_flux,
                            double tol_linear, int max_outer, int max_inner) {
    params_.tol_keff = tol_keff;
    params_.tol_flux = tol_flux;
    params_.tol_linear = tol_linear;
    params_.max_outer_iter = max_outer;
    params_.max_inner_iter = max_inner;
}

void NeutMFEM::SetBC(int attr, BCType type, double value) {
    if (attr <= 0) throw std::invalid_argument("L'attribut de bord doit être positif");
    boundary_types_[attr] = type;
    boundary_values_[attr] = value;
}

void NeutMFEM::SetRobinCoefficients(int attr, double alpha, double beta) {
    if (attr <= 0) throw std::invalid_argument("L'attribut de bord doit être positif");
    robin_alpha_[attr] = alpha;
    robin_beta_[attr] = beta;
}

void NeutMFEM::SetKrylovDimension(int dim) {
    if (dim <= 0) throw std::invalid_argument("La dimension de Krylov doit être positive");
    params_.krylov_dimension = dim;
}

mfem::IterativeSolver* NeutMFEM::CreateLinearSolver(LinearSolverType type) {
    mfem::IterativeSolver* solver = nullptr;
    switch (type) {
        case LinearSolverType::BICGSTAB:
            solver = new mfem::BiCGSTABSolver(); break;
        case LinearSolverType::GMRES: {
            auto* gmres = new mfem::GMRESSolver();
            gmres->SetKDim(params_.krylov_dimension);
            solver = gmres; break;
        }
        case LinearSolverType::MINRES:
            solver = new mfem::MINRESSolver(); break;
        case LinearSolverType::CG:
            solver = new mfem::CGSolver(); break;
        case LinearSolverType::PCG:
            solver = new mfem::CGSolver(); break;
        case LinearSolverType::FGMRES: {
            auto* fgmres = new mfem::FGMRESSolver();
            fgmres->SetKDim(params_.krylov_dimension);
            solver = fgmres; break;
        }
        default:
            throw std::runtime_error("Type de solveur non reconnu");
    }
    solver->SetRelTol(params_.tol_linear);
    solver->SetMaxIter(params_.max_inner_iter);
    solver->SetPrintLevel(verbosity_ == VerbosityLevel::DEBUG ? 1 : -1);
    return solver;
}

void NeutMFEM::ApplyQuarterRotationalSymmetry(int axis1, int axis2) {
    if (mesh_->Dimension() != 2) throw std::runtime_error("Symétrie quart: 2D seulement");
    has_quarter_symmetry_ = true; sym_axis1_ = axis1; sym_axis2_ = axis2;
    SetBC(static_cast<int>(BoundaryAttribute::LEFT_2D), BCType::MIRROR, 0.0);
    SetBC(static_cast<int>(BoundaryAttribute::BOTTOM_2D), BCType::MIRROR, 0.0);
}

void NeutMFEM::ApplyCentralSymmetry(int axis1, int axis2) {
    if (mesh_->Dimension() != 2) throw std::runtime_error("Symétrie centrale: 2D seulement");
    has_central_symmetry_ = true; central_axis1_ = axis1; central_axis2_ = axis2;
}

int NeutMFEM::GetNumElements() const { return mesh_->GetNE(); }

int NeutMFEM::GetBoundaryAttribute(int mesh_dim, int dimension, bool is_upper) const {
    if (mesh_dim == 2) {
        if (dimension == 0) return is_upper ? 2 : 1;
        if (dimension == 1) return is_upper ? 3 : 4;
    } else if (mesh_dim == 3) {
        if (dimension == 0) return is_upper ? 2 : 1;
        if (dimension == 1) return is_upper ? 4 : 3;
        if (dimension == 2) return is_upper ? 6 : 5;
    }
    throw std::runtime_error("Dimension invalide");
}

// ============================================================================
// ACCESSEURS PYTHON
// ============================================================================

py::array_t<double> NeutMFEM::GetCoefArray(mfem::GridFunction* gf,
                                          const std::vector<int>& subdiv) {
    mfem::Mesh* current_mesh = gf->FESpace()->GetMesh();
    const int dim = current_mesh->Dimension();
    const int vdim = gf->VectorDim();

    const int sx = (subdiv.size() >= 1) ? subdiv[0] : 1;
    const int sy = (subdiv.size() >= 2) ? subdiv[1] : 1;
    const int sz = (subdiv.size() >= 3) ? subdiv[2] : 1;

    const ssize_t enx = nx_ * sx;
    const ssize_t eny = (dim >= 2) ? ny_ * sy : 1;
    const ssize_t enz = (dim >= 3) ? nz_ * sz : 1;

    double* data = gf->GetData();
    const size_t d_size = sizeof(double);

    if (dim == 1) {
        return py::array_t<double>({(ssize_t)vdim, enx},
            {(ssize_t)(enx * d_size), (ssize_t)d_size}, data, py::cast(this));
    }
    if (dim == 2) {
        double* top_left = data + (eny - 1) * enx;
        return py::array_t<double>({(ssize_t)vdim, eny, enx},
            {(ssize_t)(enx * eny * d_size), -(ssize_t)(enx * d_size), (ssize_t)d_size},
            top_left, py::cast(this));
    }
    if (dim == 3) {
        double* start = data + (eny - 1) * enx;
        return py::array_t<double>({(ssize_t)vdim, enz, eny, enx},
            {(ssize_t)(enx*eny*enz*d_size), (ssize_t)(enx*eny*d_size), 
             -(ssize_t)(enx*d_size), (ssize_t)d_size}, start, py::cast(this));
    }
    throw std::runtime_error("Dimension non supportée");
}

py::array_t<double> NeutMFEM::get_D()        { return GetCoefArray(D_gf_); }
py::array_t<double> NeutMFEM::get_SRC()      { return GetCoefArray(SRC_gf_); }
py::array_t<double> NeutMFEM::get_SigR()     { return GetCoefArray(SigR_gf_); }
py::array_t<double> NeutMFEM::get_NSF()      { return GetCoefArray(NSF_gf_); }
py::array_t<double> NeutMFEM::get_KSF()      { return GetCoefArray(KSF_gf_); }
py::array_t<double> NeutMFEM::get_Chi()      { return GetCoefArray(Chi_gf_); }
py::array_t<double> NeutMFEM::get_flux()     { return GetCoefArray(Sol_Phi_gf_); }
py::array_t<double> NeutMFEM::get_flux_adj() { return GetCoefArray(Sol_Phi_adj_gf_); }

py::array_t<double> NeutMFEM::get_SigS() {
    const int dim = mesh_->Dimension();
    const int ng = n_grps_;
    double* data = SigS_gf_->GetData();
    
    if (dim == 1) {
        return py::array_t<double>({(ssize_t)ng, (ssize_t)ng, (ssize_t)nx_},
            {(ssize_t)(ng*nx_*sizeof(double)), (ssize_t)(nx_*sizeof(double)), 
             (ssize_t)sizeof(double)}, data, py::cast(this));
    }
    if (dim == 2) {
        double* top_left = data + (ny_ - 1) * nx_;
        return py::array_t<double>({(ssize_t)ng, (ssize_t)ng, (ssize_t)ny_, (ssize_t)nx_},
            {(ssize_t)(ng*nx_*ny_*sizeof(double)), (ssize_t)(nx_*ny_*sizeof(double)),
             -(ssize_t)(nx_*sizeof(double)), (ssize_t)sizeof(double)}, 
            top_left, py::cast(this));
    }
    if (dim == 3) {
        double* top_left = data + (ny_ - 1) * nx_;
        return py::array_t<double>(
            {(ssize_t)ng, (ssize_t)ng, (ssize_t)nz_, (ssize_t)ny_, (ssize_t)nx_},
            {(ssize_t)(ng*nx_*ny_*nz_*sizeof(double)), (ssize_t)(nx_*ny_*nz_*sizeof(double)),
             (ssize_t)(nx_*ny_*sizeof(double)), -(ssize_t)(nx_*sizeof(double)), 
             (ssize_t)sizeof(double)}, top_left, py::cast(this));
    }
    throw std::runtime_error("Dimension non supportée");
}

// ============================================================================
// ASSEMBLAGE DES MATRICES
// ============================================================================

void NeutMFEM::BuildMatrices() {
    Log(VerbosityLevel::NORMAL, "\n=== ASSEMBLAGE DES MATRICES ===");

    // Matrice B (Divergence)
    {
        mfem::MixedBilinearForm b_form(fes_J_, fes_Phi_);
        b_form.AddDomainIntegrator(new mfem::VectorFEDivergenceIntegrator);
        b_form.Assemble(); b_form.Finalize();
        delete B_mat_; B_mat_ = b_form.LoseMat();
    }

    const int max_bdr_attr = mesh_->bdr_attributes.Max();
    ess_bdr_J_.SetSize(max_bdr_attr); ess_bdr_Phi_.SetSize(max_bdr_attr);
    nat_bdr_J_.SetSize(max_bdr_attr);
    ess_bdr_J_ = 0; ess_bdr_Phi_ = 0; nat_bdr_J_ = 0;

    bool has_dirichlet = false, has_neumann = false, has_robin = false;
    dirichlet_values_.clear(); neumann_values_.clear();

    for (const auto& [attr, type] : boundary_types_) {
        if (attr > max_bdr_attr) continue;
        double value = boundary_values_.count(attr) ? boundary_values_[attr] : 0.0;
        switch (type) {
            case BCType::DIRICHLET:
                ess_bdr_Phi_[attr - 1] = 1; dirichlet_values_[attr] = value;
                has_dirichlet = true; break;
            case BCType::NEUMANN:
                if (std::abs(value) < 1e-12) {
                    // Neumann 0.0 (= Réflexion) -> On le traite par pénalisation forte sur J
                    if (!robin_alpha_.count(attr)) robin_alpha_[attr] = 0.0;
                    if (!robin_beta_.count(attr)) robin_beta_[attr] = 1.0e6; // Penalisation
                    robin_values_[attr] = 0.0;
                    has_robin = true; 
                    // Ne pas mettre has_neumann à true pour éviter le doublon
                } else {
                    // Neumann non-nul (Source de courant) -> Traitement standard (qui était buggé pour 0)
                    nat_bdr_J_[attr - 1] = 1; neumann_values_[attr] = value;
                    has_neumann = true;
                }
                break;
            case BCType::ROBIN: case BCType::MIRROR:
                if (!robin_alpha_.count(attr)) robin_alpha_[attr] = 0.0;
                
                if (!robin_beta_.count(attr)) {
                    if (type == BCType::MIRROR) {
                        robin_beta_[attr] = 1.0e6; // Valeur élevée pour forcer J.n = 0
                    } else {
                        robin_beta_[attr] = 1.0;
                    }
                }
                
                robin_values_[attr] = value; has_robin = true; break;
        }
    }

    ess_tdof_list_J_.SetSize(0); ess_tdof_list_Phi_.SetSize(0);
    if (has_dirichlet) {
        fes_Phi_->GetEssentialTrueDofs(ess_bdr_Phi_, ess_tdof_list_Phi_);
        dirichlet_phi_vals_.SetSize(fes_Phi_->GetVSize());
        dirichlet_phi_vals_ = 0.0;
        for (int i = 0; i < mesh_->GetNBE(); ++i) {
            int attr = mesh_->GetBdrAttribute(i);
            if (dirichlet_values_.count(attr) && ess_bdr_Phi_[attr - 1]) {
                mfem::Array<int> vdofs;
                fes_Phi_->GetBdrElementVDofs(i, vdofs);
                for (int j = 0; j < vdofs.Size(); ++j) {
                    int dof = vdofs[j];
                    double val = dirichlet_values_[attr];
                    if (dof >= 0) dirichlet_phi_vals_(dof) = val;
                    else dirichlet_phi_vals_(-1 - dof) = -val;
                }
            }
        }
    }

    // Matrices A et C par groupe
    mfem::Array<int> marker(max_bdr_attr);
    mfem::GridFunction D_temp(fes_Mat_), SigR_temp(fes_Mat_);
    mfem::GridFunction D_proj(fes_Phi_), SigR_proj(fes_Phi_);

    // Stockage des coefficients Robin pour éviter les temporaires
    std::vector<mfem::ConstantCoefficient*> robin_beta_coeffs;
    std::vector<mfem::ConstantCoefficient*> robin_alpha_coeffs;

    for (int g = 0; g < n_grps_; ++g) {
        const int offset = GetGroupOffset(g);
        D_temp.SetData(D_gf_->GetData() + offset);
        SigR_temp.SetData(SigR_gf_->GetData() + offset);
        D_proj = D_temp; SigR_proj = SigR_temp;

        InverseCoefficient invD(&D_proj);
        mfem::BilinearForm a_form(fes_J_);
        a_form.AddDomainIntegrator(new mfem::VectorFEMassIntegrator(invD));
        if (has_robin) {
            for (const auto& [attr, beta] : robin_beta_) {
                if (std::abs(beta) > 1e-14 && attr <= max_bdr_attr) {
                    marker = 0; marker[attr - 1] = 1;
                    auto* beta_coeff = new mfem::ConstantCoefficient(beta);
                    robin_beta_coeffs.push_back(beta_coeff);
                    a_form.AddBoundaryIntegrator(
                        new mfem::MassIntegrator(*beta_coeff), marker);
                }
            }
        }
        a_form.Assemble(); a_form.Finalize();
        delete A_mats_[g]; A_mats_[g] = a_form.LoseMat();

        mfem::GridFunctionCoefficient sigR_coeff(&SigR_proj);
        mfem::BilinearForm c_form(fes_Phi_);
        c_form.AddDomainIntegrator(new mfem::MassIntegrator(sigR_coeff));
        if (has_robin) {
            for (const auto& [attr, alpha] : robin_alpha_) {
                if (std::abs(alpha) > 1e-14 && attr <= max_bdr_attr) {
                    marker = 0; marker[attr - 1] = 1;
                    auto* alpha_coeff = new mfem::ConstantCoefficient(alpha);
                    robin_alpha_coeffs.push_back(alpha_coeff);
                    c_form.AddBoundaryIntegrator(
                        new mfem::MassIntegrator(*alpha_coeff), marker);
                }
            }
        }
        c_form.Assemble(); c_form.Finalize();
        delete C_mats_[g]; C_mats_[g] = c_form.LoseMat();
    }

    // Nettoyage des coefficients temporaires
    for (auto* c : robin_beta_coeffs) delete c;
    for (auto* c : robin_alpha_coeffs) delete c;

    delete BT_mat_; BT_mat_ = mfem::Transpose(*B_mat_);

    // Matrices de fission
    mfem::GridFunction NSF_proj(fes_Phi_);
    for (int g = 0; g < n_grps_; ++g) {
        mfem::GridFunction NSF_temp(fes_Mat_);
        NSF_temp.SetData(NSF_gf_->GetData() + GetGroupOffset(g));
        NSF_proj = NSF_temp;
        mfem::GridFunctionCoefficient NSF_coeff(&NSF_proj);
        mfem::BilinearForm f_form(fes_Phi_);
        f_form.AddDomainIntegrator(new mfem::MassIntegrator(NSF_coeff));
        f_form.Assemble(); f_form.Finalize();
        delete M_fiss_[g]; M_fiss_[g] = f_form.LoseMat();
    }

    // Matrices de scattering
    mfem::GridFunction SigS_proj(fes_Phi_);
    for (int g_from = 0; g_from < n_grps_; ++g_from) {
        for (int g_to = 0; g_to < n_grps_; ++g_to) {
            if (g_from == g_to) continue;
            const int idx = g_to * n_grps_ + g_from;
            mfem::GridFunction sigs_temp(fes_Mat_);
            sigs_temp.SetData(SigS_gf_->GetData() + GetSigSOffset(g_from, g_to));
            if (sigs_temp.Norml2() > 1e-14) {
                SigS_proj = sigs_temp;
                mfem::GridFunctionCoefficient sigs_coeff(&SigS_proj);
                mfem::BilinearForm s_form(fes_Phi_);
                s_form.AddDomainIntegrator(new mfem::MassIntegrator(sigs_coeff));
                s_form.Assemble(); s_form.Finalize();
                delete M_scatter_[idx]; M_scatter_[idx] = s_form.LoseMat();
            }
        }
    }

    // Formes linéaires BC
    if (has_neumann || has_robin) {
        neumann_lf_J_.resize(n_grps_);
        robin_lf_Phi_.resize(n_grps_);
        mfem::Array<int> bdr_marker(max_bdr_attr);
        
        // Stockage des coefficients pour éviter les temporaires
        std::vector<mfem::ConstantCoefficient*> neumann_coeffs;
        std::vector<mfem::ConstantCoefficient*> robin_val_coeffs;
        
        for (int g = 0; g < n_grps_; ++g) {
            if (has_neumann) {
                neumann_lf_J_[g] = new mfem::LinearForm(fes_J_);
                for (const auto& [attr, val] : neumann_values_) {
                    if (nat_bdr_J_[attr - 1] && attr <= max_bdr_attr) {
                        bdr_marker = 0; bdr_marker[attr - 1] = 1;
                        auto* val_coeff = new mfem::ConstantCoefficient(val);
                        neumann_coeffs.push_back(val_coeff);
                        neumann_lf_J_[g]->AddBoundaryIntegrator(
                            new mfem::VectorFEBoundaryFluxLFIntegrator(*val_coeff), bdr_marker);
                    }
                }
                neumann_lf_J_[g]->Assemble();
            }
            if (has_robin) {
                robin_lf_Phi_[g] = new mfem::LinearForm(fes_Phi_);
                for (const auto& [attr, val] : robin_values_) {
                    if (attr <= max_bdr_attr) {
                        bdr_marker = 0; bdr_marker[attr - 1] = 1;
                        auto* val_coeff = new mfem::ConstantCoefficient(val);
                        robin_val_coeffs.push_back(val_coeff);
                        robin_lf_Phi_[g]->AddBoundaryIntegrator(
                            new mfem::BoundaryLFIntegrator(*val_coeff), bdr_marker);
                    }
                }
                robin_lf_Phi_[g]->Assemble();
            }
        }
        
        // Note: Les coefficients sont gardés en mémoire car les LinearForm 
        // peuvent en avoir besoin après l'assemblage pour des évaluations futures
        // Dans une version plus propre, ils seraient membres de la classe
    }

    // ========================================================================
    // FORME CONDENSÉE : On n'assemble PAS K explicitement
    // On utilise un opérateur implicite dans SolveGroupCondensed
    // ========================================================================
    if (use_condensed_form_) {
        Log(VerbosityLevel::NORMAL, "Mode condensé activé (opérateur implicite)");
        // Les matrices K_condensed_ restent nullptr
        // L'opérateur K = B*A^{-1}*B^T + C sera appliqué à la volée
    }

    Log(VerbosityLevel::NORMAL, "Assemblage terminé\n");
}

void NeutMFEM::ExtractGroupFromMultiGroup(mfem::GridFunction* multi_gf, int g,
                                         mfem::GridFunction* single_gf) {
    const int n = multi_gf->Size() / n_grps_;
    std::memcpy(single_gf->GetData(), multi_gf->GetData() + g * n, n * sizeof(double));
}

void NeutMFEM::InjectGroupIntoMultiGroup(mfem::GridFunction* single_gf, int g,
                                        mfem::GridFunction* multi_gf) {
    const int n = single_gf->Size();
    std::memcpy(multi_gf->GetData() + g * n, single_gf->GetData(), n * sizeof(double));
}

// ============================================================================
// RÉSOLUTION D'UN GROUPE INDIVIDUEL
// ============================================================================

void NeutMFEM::SolveGroupInternal(int g_idx, const mfem::Vector& integrated_src,
                                 LinearSolverType type) {
    const int nJ = fes_J_->GetVSize();
    const int nPhi = fes_Phi_->GetVSize();
    const int mesh_dim = mesh_->Dimension();

    mfem::Vector j_g, phi_g;
    j_g.SetDataAndSize(Sol_J_gf_->GetData() + g_idx * nJ, nJ);
    phi_g.SetDataAndSize(Sol_Phi_gf_->GetData() + g_idx * nPhi, nPhi);

    mfem::Array<int> offsets(3);
    offsets[0] = 0; offsets[1] = nJ; offsets[2] = nPhi;
    offsets.PartialSum();

    mfem::BlockVector rhs(offsets), x(offsets);
    rhs = 0.0;
    x.GetBlock(0) = j_g;
    x.GetBlock(1) = phi_g;
    rhs.GetBlock(1) = integrated_src;

    update_refl(*Sol_Phi_gf_, g_idx);

    if (!active_reflectors_.empty()) {
        for (const auto& kv : active_reflectors_) {
            const auto& key = kv.first;
            const int dimension = key.first;
            const bool is_upper = (key.second == 1);
            const mfem::Vector& rhs_refl = reflector_rhs_.at(key);
            const double b_g = rhs_refl(g_idx);
            if (std::abs(b_g) < 1e-14) continue;

            const int target_attr = GetBoundaryAttribute(mesh_dim, dimension, is_upper);
            mfem::Vector boundary_rhs(nPhi);
            boundary_rhs = 0.0;

            for (int i = 0; i < mesh_->GetNBE(); ++i) {
                if (mesh_->GetBdrAttribute(i) != target_attr) continue;
                auto* T = mesh_->GetBdrFaceTransformations(i);
                if (!T) continue;

                mfem::Array<int> vdofs;
                fes_Phi_->GetBdrElementVDofs(i, vdofs);
                const auto* ir = &mfem::IntRules.Get(
                    mesh_->GetBdrElement(i)->GetGeometryType(),
                    2 * fes_Phi_->GetMaxElementOrder() + 2);
                const auto* fe = fes_Phi_->GetBE(i);
                mfem::Vector loc_rhs(vdofs.Size());
                loc_rhs = 0.0;

                for (int j = 0; j < ir->GetNPoints(); ++j) {
                    const auto& ip = ir->IntPoint(j);
                    T->SetAllIntPoints(&ip);
                    mfem::Vector shape(vdofs.Size());
                    fe->CalcShape(ip, shape);
                    const double weight = ip.weight * T->Weight() * b_g;
                    loc_rhs.Add(weight, shape);
                }

                for (int k = 0; k < vdofs.Size(); ++k) {
                    const int dof = vdofs[k];
                    if (dof >= 0) boundary_rhs(dof) += loc_rhs(k);
                    else boundary_rhs(-1 - dof) -= loc_rhs(k);
                }
            }
            rhs.GetBlock(1) += boundary_rhs;
        }
    }

    mfem::BlockOperator op(offsets);
    op.SetBlock(0, 0, A_mats_[g_idx]);
    op.SetBlock(0, 1, BT_mat_, -1.0);
    op.SetBlock(1, 0, B_mat_, -1.0);
    op.SetBlock(1, 1, C_mats_[g_idx]);

    std::unique_ptr<mfem::IterativeSolver> solver(CreateLinearSolver(type));
    solver->SetOperator(op);
    solver->Mult(rhs, x);

    j_g = x.GetBlock(0);
    phi_g = x.GetBlock(1);
}

// ============================================================================
// RÉSOLUTION D'UN GROUPE - FORME CONDENSÉE
// ============================================================================

void NeutMFEM::SolveGroupCondensed(int g_idx, const mfem::Vector& integrated_src,
                                   LinearSolverType type) {
    const int nPhi = fes_Phi_->GetVSize();
    const int mesh_dim = mesh_->Dimension();

    mfem::Vector phi_g;
    phi_g.SetDataAndSize(Sol_Phi_gf_->GetData() + g_idx * nPhi, nPhi);

    // Préparer le second membre
    mfem::Vector rhs(nPhi);
    rhs = integrated_src;

    // Traitement des réflecteurs (même logique que forme mixte)
    update_refl(*Sol_Phi_gf_, g_idx);

    if (!active_reflectors_.empty()) {
        for (const auto& kv : active_reflectors_) {
            const auto& key = kv.first;
            const int dimension = key.first;
            const bool is_upper = (key.second == 1);
            const mfem::Vector& rhs_refl = reflector_rhs_.at(key);
            const double b_g = rhs_refl(g_idx);
            if (std::abs(b_g) < 1e-14) continue;

            const int target_attr = GetBoundaryAttribute(mesh_dim, dimension, is_upper);
            mfem::Vector boundary_rhs(nPhi);
            boundary_rhs = 0.0;

            for (int i = 0; i < mesh_->GetNBE(); ++i) {
                if (mesh_->GetBdrAttribute(i) != target_attr) continue;
                auto* T = mesh_->GetBdrFaceTransformations(i);
                if (!T) continue;

                mfem::Array<int> vdofs;
                fes_Phi_->GetBdrElementVDofs(i, vdofs);
                const auto* ir = &mfem::IntRules.Get(
                    mesh_->GetBdrElement(i)->GetGeometryType(),
                    2 * fes_Phi_->GetMaxElementOrder() + 2);
                const auto* fe = fes_Phi_->GetBE(i);
                mfem::Vector loc_rhs(vdofs.Size());
                loc_rhs = 0.0;

                for (int j = 0; j < ir->GetNPoints(); ++j) {
                    const auto& ip = ir->IntPoint(j);
                    T->SetAllIntPoints(&ip);
                    mfem::Vector shape(vdofs.Size());
                    fe->CalcShape(ip, shape);
                    const double weight = ip.weight * T->Weight() * b_g;
                    loc_rhs.Add(weight, shape);
                }

                for (int k = 0; k < vdofs.Size(); ++k) {
                    const int dof = vdofs[k];
                    if (dof >= 0) boundary_rhs(dof) += loc_rhs(k);
                    else boundary_rhs(-1 - dof) -= loc_rhs(k);
                }
            }
            rhs += boundary_rhs;
        }
    }

    // Créer l'opérateur condensé implicite K = B * A^{-1} * B^T + C
    CondensedOperator K_op(A_mats_[g_idx], B_mat_, BT_mat_, C_mats_[g_idx],
                           params_.tol_linear, params_.max_inner_iter);
    
    // Résolution K * phi_g = rhs
    // Note: pas de préconditionneur simple car K_op n'est pas une SparseMatrix
    // Le préconditionneur interne de CondensedOperator (pour A^{-1}) suffit généralement
    std::unique_ptr<mfem::IterativeSolver> solver(CreateLinearSolver(type));
    solver->SetOperator(K_op);
    
    mfem::Vector sol(nPhi);
    sol = phi_g;  // Initialisation avec la solution précédente
    solver->Mult(rhs, sol);
    
    phi_g = sol;
}

// ============================================================================
// RECONSTRUCTION DU COURANT (POST-TRAITEMENT)
// ============================================================================

void NeutMFEM::ReconstructCurrent(int g_idx) {
    // J = -D * grad(Phi)
    // En forme faible mixte: A * J = B^T * Phi
    // Donc: J = A^{-1} * B^T * Phi
    
    const int nJ = fes_J_->GetVSize();
    const int nPhi = fes_Phi_->GetVSize();
    
    mfem::Vector phi_g, j_g;
    phi_g.SetDataAndSize(Sol_Phi_gf_->GetData() + g_idx * nPhi, nPhi);
    j_g.SetDataAndSize(Sol_J_gf_->GetData() + g_idx * nJ, nJ);
    
    // Calcul de B^T * phi_g
    mfem::Vector BT_phi(nJ);
    BT_mat_->Mult(phi_g, BT_phi);
    
    // Résolution A * j_g = BT_phi
    //mfem::DSmoother prec(*A_mats_[g_idx], 1);
    mfem::CGSolver cg;
    cg.SetOperator(*A_mats_[g_idx]);
    //cg.SetPreconditioner(prec);
    cg.SetRelTol(1e-5);
    cg.SetMaxIter(200);
    cg.SetPrintLevel(-1);
    
    j_g = 0.0;
    cg.Mult(BT_phi, j_g);
}

// ============================================================================
// RÉSOLUTION GROUPE ADJOINT - FORME CONDENSÉE
// ============================================================================

void NeutMFEM::SolveAdjointGroupCondensed(int g_idx, const mfem::Vector& integrated_src,
                                          LinearSolverType type) {
    const int nPhi = fes_Phi_->GetVSize();

    mfem::Vector phi_adj_g;
    phi_adj_g.SetDataAndSize(Sol_Phi_adj_gf_->GetData() + g_idx * nPhi, nPhi);

    // Pour l'adjoint, K^T = K (matrices symétriques en diffusion)
    // Donc on résout le même système
    
    mfem::Vector rhs(nPhi);
    rhs = integrated_src;

    // Créer l'opérateur condensé implicite
    CondensedOperator K_op(A_mats_[g_idx], B_mat_, BT_mat_, C_mats_[g_idx],
                           params_.tol_linear, params_.max_inner_iter);
    
    std::unique_ptr<mfem::IterativeSolver> solver(CreateLinearSolver(type));
    solver->SetOperator(K_op);
    
    mfem::Vector sol(nPhi);
    sol = phi_adj_g;
    solver->Mult(rhs, sol);
    
    phi_adj_g = sol;
}

// ============================================================================
// RÉSOLUTION GROUPE ADJOINT
// ============================================================================

void NeutMFEM::SolveAdjointGroupInternal(int g_idx, const mfem::Vector& integrated_src,
                                        LinearSolverType type) {
    const int nJ = fes_J_->GetVSize();
    const int nPhi = fes_Phi_->GetVSize();

    mfem::Vector j_adj_g, phi_adj_g;
    j_adj_g.SetDataAndSize(Sol_J_adj_gf_->GetData() + g_idx * nJ, nJ);
    phi_adj_g.SetDataAndSize(Sol_Phi_adj_gf_->GetData() + g_idx * nPhi, nPhi);

    mfem::Array<int> offsets(3);
    offsets[0] = 0; offsets[1] = nJ; offsets[2] = nPhi;
    offsets.PartialSum();

    mfem::BlockVector rhs(offsets), x(offsets);
    rhs = 0.0;
    x.GetBlock(0) = j_adj_g;
    x.GetBlock(1) = phi_adj_g;
    rhs.GetBlock(1) = integrated_src;

    // Opérateur adjoint (transposé)
    mfem::BlockOperator op(offsets);
    op.SetBlock(0, 0, A_mats_[g_idx]);
    op.SetBlock(0, 1, BT_mat_, -1.0);
    op.SetBlock(1, 0, B_mat_, -1.0);
    op.SetBlock(1, 1, C_mats_[g_idx]);

    std::unique_ptr<mfem::IterativeSolver> solver(CreateLinearSolver(type));
    solver->SetOperator(op);
    solver->Mult(rhs, x);

    j_adj_g = x.GetBlock(0);
    phi_adj_g = x.GetBlock(1);
}

// ============================================================================
// HOMOGÉNÉISATION AVEC FACTEURS X/Y/Z SÉPARÉS
// ============================================================================

void NeutMFEM::HomogenizeToCoarse(const mfem::GridFunction& fine_gf,
                                 const CoarseFactors& factors,
                                 std::vector<double>& coarse_values,
                                 bool volume_weighted) const {
    const int dim = mesh_->Dimension();
    const int fx = factors.x;
    const int fy = factors.y;
    const int fz = factors.z;

    const int nx_coarse = (nx_ + fx - 1) / fx;
    const int ny_coarse = (dim >= 2) ? (ny_ + fy - 1) / fy : 1;
    const int nz_coarse = (dim == 3) ? (nz_ + fz - 1) / fz : 1;

    const int n_coarse = nx_coarse * ny_coarse * nz_coarse;
    coarse_values.assign(n_coarse, 0.0);
    std::vector<double> coarse_volume(n_coarse, 0.0);

    for (int i_fine = 0; i_fine < mesh_->GetNE(); ++i_fine) {
        const int ix_fine = i_fine % nx_;
        const int iy_fine = (dim >= 2) ? (i_fine / nx_) % ny_ : 0;
        const int iz_fine = (dim == 3) ? i_fine / (nx_ * ny_) : 0;

        const int ix_coarse = ix_fine / fx;
        const int iy_coarse = iy_fine / fy;
        const int iz_coarse = iz_fine / fz;

        const int i_coarse = ix_coarse + iy_coarse * nx_coarse + 
                             iz_coarse * nx_coarse * ny_coarse;

        const double vol = mesh_->GetElementVolume(i_fine);
        const double val = fine_gf(i_fine);

        if (volume_weighted) {
            coarse_values[i_coarse] += val * vol;
            coarse_volume[i_coarse] += vol;
        } else {
            coarse_values[i_coarse] += val;
            coarse_volume[i_coarse] += 1.0;
        }
    }

    for (int c = 0; c < n_coarse; ++c) {
        if (coarse_volume[c] > 0.0) {
            coarse_values[c] /= coarse_volume[c];
        }
    }
}

void NeutMFEM::InterpolateFromCoarse(const std::vector<double>& coarse_values,
                                    const CoarseFactors& factors,
                                    mfem::GridFunction& fine_gf) const {
    const int dim = mesh_->Dimension();
    const int fx = factors.x;
    const int fy = factors.y;
    const int fz = factors.z;

    const int nx_coarse = (nx_ + fx - 1) / fx;
    const int ny_coarse = (dim >= 2) ? (ny_ + fy - 1) / fy : 1;

    for (int i_fine = 0; i_fine < mesh_->GetNE(); ++i_fine) {
        const int ix_fine = i_fine % nx_;
        const int iy_fine = (dim >= 2) ? (i_fine / nx_) % ny_ : 0;
        const int iz_fine = (dim == 3) ? i_fine / (nx_ * ny_) : 0;

        const int ix_coarse = ix_fine / fx;
        const int iy_coarse = iy_fine / fy;
        const int iz_coarse = iz_fine / fz;

        const int i_coarse = ix_coarse + iy_coarse * nx_coarse + 
                             iz_coarse * nx_coarse * ny_coarse;

        fine_gf(i_fine) = coarse_values[i_coarse];
    }
}

std::vector<double> NeutMFEM::ComputeCoarseBreaks(const std::vector<double>& fine_breaks,
                                                  int coarse_factor) const {
    std::vector<double> coarse_breaks;
    coarse_breaks.push_back(fine_breaks[0]);
    for (size_t i = coarse_factor; i < fine_breaks.size(); i += coarse_factor) {
        coarse_breaks.push_back(fine_breaks[i]);
    }
    if (coarse_breaks.back() != fine_breaks.back()) {
        coarse_breaks.push_back(fine_breaks.back());
    }
    return coarse_breaks;
}

// ============================================================================
// RÉSOLUTION K-EFF (VERSION SIMPLE)
// ============================================================================

double NeutMFEM::SolveKeff(LinearSolverType solver_type) {
    return SolveKeff(solver_type, false, CoarseFactors(1, 1, 1), 30, 1e-3);
}

// ============================================================================
// RÉSOLUTION K-EFF (VERSION COMPLÈTE AVEC COARSE X/Y/Z)
// ============================================================================

double NeutMFEM::SolveKeff(LinearSolverType solver_type, bool use_coarse_init,
                          const CoarseFactors& coarse_factors,
                          int coarse_max_iter, double coarse_tol_keff) {
    Log(VerbosityLevel::NORMAL, "\n=== CALCUL DE K-EFF ===");

    const int ng = n_grps_;
    const int n_phi_dofs = fes_Phi_->GetVSize();
    const int dim = mesh_->Dimension();

    double keff = 1.0;
    double diff_k = 1.0;

    // ========================================================================
    // PHASE COARSE (optionnelle)
    // ========================================================================
    if (use_coarse_init && (coarse_factors.x > 1 || coarse_factors.y > 1 || coarse_factors.z > 1)) {
        Log(VerbosityLevel::NORMAL, "\n--- Phase d'initialisation coarse ---");
        Log(VerbosityLevel::NORMAL, "Facteurs: ", coarse_factors.x, " × ", 
            coarse_factors.y, " × ", coarse_factors.z);

        // Extraction des breaks
        std::set<double> x_set, y_set, z_set;
        for (int i = 0; i < mesh_->GetNV(); ++i) {
            double* v = mesh_->GetVertex(i);
            x_set.insert(v[0]);
            if (dim >= 2) y_set.insert(v[1]);
            if (dim == 3) z_set.insert(v[2]);
        }

        std::vector<double> x_fine(x_set.begin(), x_set.end());
        std::vector<double> y_fine = (dim >= 2) ? std::vector<double>(y_set.begin(), y_set.end()) 
                                                : std::vector<double>{0.0, 1.0};
        std::vector<double> z_fine = (dim == 3) ? std::vector<double>(z_set.begin(), z_set.end()) 
                                                : std::vector<double>{0.0};

        auto x_coarse = ComputeCoarseBreaks(x_fine, coarse_factors.x);
        auto y_coarse = (dim >= 2) ? ComputeCoarseBreaks(y_fine, coarse_factors.y) 
                                   : std::vector<double>{0.0};
        auto z_coarse = (dim == 3) ? ComputeCoarseBreaks(z_fine, coarse_factors.z) 
                                   : std::vector<double>{0.0};

        const int nx_c = x_coarse.size() - 1;
        const int ny_c = (dim >= 2) ? y_coarse.size() - 1 : 1;
        const int nz_c = (dim == 3) ? z_coarse.size() - 1 : 1;

        Log(VerbosityLevel::NORMAL, "Maillage fin: ", nx_, " × ", ny_, " × ", nz_);
        Log(VerbosityLevel::NORMAL, "Maillage coarse: ", nx_c, " × ", ny_c, " × ", nz_c);

        std::unique_ptr<NeutMFEM> coarse_solver(new NeutMFEM(0, ng, x_coarse, y_coarse, z_coarse));
        coarse_solver->SetVerbosity(VerbosityLevel::LIGHT);

        // Homogénéisation des sections efficaces
        std::vector<double> coarse_vals;
        const int n_mat_c = coarse_solver->fes_Mat_->GetVSize();

        for (int g = 0; g < ng; ++g) {
            const int off_fine = g * mesh_->GetNE();
            const int off_coarse = g * n_mat_c;

            mfem::GridFunction D_g(fes_Mat_); D_g.SetData(D_gf_->GetData() + off_fine);
            HomogenizeToCoarse(D_g, coarse_factors, coarse_vals, true);
            for (int i = 0; i < n_mat_c; ++i)
                (*coarse_solver->D_gf_)(off_coarse + i) = coarse_vals[i % coarse_vals.size()];

            mfem::GridFunction SigR_g(fes_Mat_); SigR_g.SetData(SigR_gf_->GetData() + off_fine);
            HomogenizeToCoarse(SigR_g, coarse_factors, coarse_vals, true);
            for (int i = 0; i < n_mat_c; ++i)
                (*coarse_solver->SigR_gf_)(off_coarse + i) = coarse_vals[i % coarse_vals.size()];

            mfem::GridFunction NSF_g(fes_Mat_); NSF_g.SetData(NSF_gf_->GetData() + off_fine);
            HomogenizeToCoarse(NSF_g, coarse_factors, coarse_vals, true);
            for (int i = 0; i < n_mat_c; ++i)
                (*coarse_solver->NSF_gf_)(off_coarse + i) = coarse_vals[i % coarse_vals.size()];

            mfem::GridFunction Chi_g(fes_Mat_); Chi_g.SetData(Chi_gf_->GetData() + off_fine);
            HomogenizeToCoarse(Chi_g, coarse_factors, coarse_vals, true);
            for (int i = 0; i < n_mat_c; ++i)
                (*coarse_solver->Chi_gf_)(off_coarse + i) = coarse_vals[i % coarse_vals.size()];
        }

        // Scattering
        for (int g_from = 0; g_from < ng; ++g_from) {
            for (int g_to = 0; g_to < ng; ++g_to) {
                const int off_fine = (g_to * ng + g_from) * mesh_->GetNE();
                const int off_coarse = (g_to * ng + g_from) * n_mat_c;
                mfem::GridFunction SigS_pair(fes_Mat_);
                SigS_pair.SetData(SigS_gf_->GetData() + off_fine);
                HomogenizeToCoarse(SigS_pair, coarse_factors, coarse_vals, true);
                for (int i = 0; i < n_mat_c; ++i)
                    (*coarse_solver->SigS_gf_)(off_coarse + i) = coarse_vals[i % coarse_vals.size()];
            }
        }

        // BC coarse
        for (const auto& [attr, type] : boundary_types_) {
            double val = boundary_values_.count(attr) ? boundary_values_[attr] : 0.0;
            coarse_solver->SetBC(attr, type, val);
        }
        for (const auto& [attr, alpha] : robin_alpha_) {
            double beta = robin_beta_.count(attr) ? robin_beta_[attr] : 1.0;
            coarse_solver->SetRobinCoefficients(attr, alpha, beta);
        }

        coarse_solver->BuildMatrices();
        coarse_solver->SetTolerances(coarse_tol_keff, 1e-4, 1e-4, coarse_max_iter, 100);

        const int n_phi_c = coarse_solver->fes_Phi_->GetVSize();
        mfem::Vector total_fiss_c(n_phi_c), prod_c(n_phi_c);
        mfem::Vector phi_g_c, phi_gp_c;

        *coarse_solver->Sol_Phi_gf_ = 1.0;
        *coarse_solver->Sol_Phi_gf_ /= coarse_solver->Sol_Phi_gf_->Norml2();

        double diff_k_c = 1.0;
        for (int it = 0; it < coarse_max_iter && diff_k_c > coarse_tol_keff; ++it) {
            total_fiss_c = 0.0;
            for (int g = 0; g < ng; ++g) {
                phi_g_c.SetDataAndSize(coarse_solver->Sol_Phi_gf_->GetData() + g * n_phi_c, n_phi_c);
                coarse_solver->M_fiss_[g]->AddMult(phi_g_c, total_fiss_c);
            }
            double prod_old = total_fiss_c.Sum();
            total_fiss_c *= (1.0 / keff);

            for (int g = 0; g < ng; ++g) {
                mfem::Vector group_rhs(n_phi_c);
                group_rhs = 0.0;
                for (int i = 0; i < n_phi_c; ++i) {
                    int mat_i = i % n_mat_c;
                    group_rhs(i) = (*coarse_solver->Chi_gf_)(g * n_mat_c + mat_i) * total_fiss_c(i);
                }
                for (int gp = 0; gp < ng; ++gp) {
                    if (g == gp) continue;
                    int idx = g * ng + gp;
                    if (!coarse_solver->M_scatter_[idx]) continue;
                    phi_gp_c.SetDataAndSize(coarse_solver->Sol_Phi_gf_->GetData() + gp * n_phi_c, n_phi_c);
                    coarse_solver->M_scatter_[idx]->AddMult(phi_gp_c, group_rhs);
                }
                
                // Utiliser forme condensée ou mixte selon la configuration
                if (coarse_solver->use_condensed_form_) {
                    coarse_solver->SolveGroupCondensed(g, group_rhs, solver_type);
                } else {
                    coarse_solver->SolveGroupInternal(g, group_rhs, solver_type);
                }
            }

            double prod_new = 0.0;
            for (int g = 0; g < ng; ++g) {
                phi_g_c.SetDataAndSize(coarse_solver->Sol_Phi_gf_->GetData() + g * n_phi_c, n_phi_c);
                coarse_solver->M_fiss_[g]->Mult(phi_g_c, prod_c);
                prod_new += prod_c.Sum();
            }

            double keff_new = keff * (prod_new / prod_old);
            diff_k_c = std::abs(keff_new - keff);
            keff = keff_new;
            *coarse_solver->Sol_Phi_gf_ /= coarse_solver->Sol_Phi_gf_->Norml2();

            if (verbosity_ >= VerbosityLevel::LIGHT && (it % 5 == 0 || diff_k_c < coarse_tol_keff)) {
                std::cout << "  [Coarse] It " << std::setw(3) << it
                          << ": k=" << std::fixed << std::setprecision(6) << keff
                          << " (dk=" << std::scientific << std::setprecision(2) << diff_k_c << ")\n";
            }
        }

        // ====================================================================
        // STOCKAGE DES DONNÉES COARSE POUR EXPORT VTK
        // ====================================================================
        coarse_factors_used_ = coarse_factors;
        nx_coarse_ = nx_c;
        ny_coarse_ = ny_c;
        nz_coarse_ = nz_c;

        // Nettoyage des anciennes données coarse
        delete mesh_coarse_; delete fes_Mat_coarse_; delete fes_Phi_coarse_;
        delete Sol_Phi_coarse_gf_; delete D_coarse_gf_; delete SigR_coarse_gf_;
        delete NSF_coarse_gf_; delete Chi_coarse_gf_;
        delete SigS_coarse_gf_; delete SRC_coarse_gf_;

        // Création du maillage coarse persistant
        if (dim == 3) {
            mesh_coarse_ = new mfem::Mesh(mfem::Mesh::MakeCartesian3D(
                nx_c, ny_c, nz_c, mfem::Element::HEXAHEDRON, 1.0, 1.0, 1.0, false));
        } else if (dim == 2) {
            mesh_coarse_ = new mfem::Mesh(mfem::Mesh::MakeCartesian2D(
                nx_c, ny_c, mfem::Element::QUADRILATERAL, false, 1.0, 1.0, false));
        } else {
            mesh_coarse_ = new mfem::Mesh(mfem::Mesh::MakeCartesian1D(nx_c, 1.0));
        }

        // Mise à jour des coordonnées du maillage coarse
        for (int i = 0; i < mesh_coarse_->GetNV(); ++i) {
            double* v = mesh_coarse_->GetVertex(i);
            int ix = std::min(static_cast<int>(std::round(v[0] * nx_c)), nx_c);
            v[0] = x_coarse[ix];
            if (dim >= 2) {
                int iy = std::min(static_cast<int>(std::round(v[1] * ny_c)), ny_c);
                v[1] = y_coarse[iy];
            }
            if (dim == 3) {
                int iz = std::min(static_cast<int>(std::round(v[2] * nz_c)), nz_c);
                v[2] = z_coarse[iz];
            }
        }
        mesh_coarse_->Finalize(true);

        // Espaces EF coarse
        auto* fec_mat_c = new mfem::L2_FECollection(0, dim);
        auto* fec_phi_c = new mfem::L2_FECollection(0, dim);
        fes_Mat_coarse_ = new mfem::FiniteElementSpace(mesh_coarse_, fec_mat_c);
        fes_Phi_coarse_ = new mfem::FiniteElementSpace(mesh_coarse_, fec_phi_c);

        // Copie des GridFunctions coarse
        const int n_elem_c = nx_c * ny_c * nz_c;
        
        Sol_Phi_coarse_gf_ = new mfem::GridFunction(fes_Phi_coarse_);
        Sol_Phi_coarse_gf_->SetSize(ng * n_elem_c);
        for (int i = 0; i < ng * n_phi_c; ++i)
            (*Sol_Phi_coarse_gf_)(i) = (*coarse_solver->Sol_Phi_gf_)(i);

        D_coarse_gf_ = new mfem::GridFunction(fes_Mat_coarse_);
        D_coarse_gf_->SetSize(ng * n_mat_c);
        for (int i = 0; i < ng * n_mat_c; ++i)
            (*D_coarse_gf_)(i) = (*coarse_solver->D_gf_)(i);

        SigR_coarse_gf_ = new mfem::GridFunction(fes_Mat_coarse_);
        SigR_coarse_gf_->SetSize(ng * n_mat_c);
        for (int i = 0; i < ng * n_mat_c; ++i)
            (*SigR_coarse_gf_)(i) = (*coarse_solver->SigR_gf_)(i);

        NSF_coarse_gf_ = new mfem::GridFunction(fes_Mat_coarse_);
        NSF_coarse_gf_->SetSize(ng * n_mat_c);
        for (int i = 0; i < ng * n_mat_c; ++i)
            (*NSF_coarse_gf_)(i) = (*coarse_solver->NSF_gf_)(i);

        Chi_coarse_gf_ = new mfem::GridFunction(fes_Mat_coarse_);
        Chi_coarse_gf_->SetSize(ng * n_mat_c);
        for (int i = 0; i < ng * n_mat_c; ++i)
            (*Chi_coarse_gf_)(i) = (*coarse_solver->Chi_gf_)(i);

        SigS_coarse_gf_ = new mfem::GridFunction(fes_Mat_coarse_);
        SigS_coarse_gf_->SetSize(ng * ng * n_mat_c);
        for (int i = 0; i < ng * ng * n_mat_c; ++i)
            (*SigS_coarse_gf_)(i) = (*coarse_solver->SigS_gf_)(i);

        SRC_coarse_gf_ = new mfem::GridFunction(fes_Mat_coarse_);
        SRC_coarse_gf_->SetSize(ng * n_mat_c);
        for (int i = 0; i < ng * n_mat_c; ++i)
            (*SRC_coarse_gf_)(i) = (*coarse_solver->SRC_gf_)(i);

        Log(VerbosityLevel::DEBUG, "Données coarse stockées pour export VTK");

        // Interpolation vers maillage fin
        for (int g = 0; g < ng; ++g) {
            const int n_elem_c = nx_c * ny_c * nz_c;
            std::vector<double> phi_c_g(n_elem_c);
            for (int i = 0; i < n_elem_c; ++i)
                phi_c_g[i] = (*coarse_solver->Sol_Phi_gf_)(g * n_phi_c + i);

            mfem::GridFunction phi_fine_g(fes_Phi_);
            phi_fine_g.SetData(Sol_Phi_gf_->GetData() + g * n_phi_dofs);
            InterpolateFromCoarse(phi_c_g, coarse_factors, phi_fine_g);
        }
        *Sol_Phi_gf_ /= Sol_Phi_gf_->Norml2();

        Log(VerbosityLevel::NORMAL, "Phase coarse terminée, k-eff initial: ", keff);
        Log(VerbosityLevel::NORMAL, "\n--- Phase de raffinement fine ---");
    }

    // ========================================================================
    // PHASE FINE
    // ========================================================================
    ChebyshevAccelerator accel(15, 0.98);
    mfem::Vector total_fiss(n_phi_dofs), prod_g(n_phi_dofs);
    mfem::Vector phi_g_view, phi_gp_view;

    diff_k = 1.0;
    for (int it = 0; it < params_.max_outer_iter && diff_k > params_.tol_keff; ++it) {
        total_fiss = 0.0;
        for (int g = 0; g < ng; ++g) {
            phi_g_view.SetDataAndSize(Sol_Phi_gf_->GetData() + g * n_phi_dofs, n_phi_dofs);
            M_fiss_[g]->AddMult(phi_g_view, total_fiss);
        }
        double prod_old = total_fiss.Sum();
        total_fiss *= (1.0 / keff);

        for (int g = 0; g < ng; ++g) {
            mfem::Vector group_rhs(n_phi_dofs);
            group_rhs = 0.0;
            for (int i = 0; i < n_phi_dofs; ++i)
                group_rhs(i) = (*Chi_gf_)(g * n_phi_dofs + i) * total_fiss(i);

            for (int gp = 0; gp < ng; ++gp) {
                if (g == gp) continue;
                int idx = g * ng + gp;
                if (!M_scatter_[idx]) continue;
                phi_gp_view.SetDataAndSize(Sol_Phi_gf_->GetData() + gp * n_phi_dofs, n_phi_dofs);
                M_scatter_[idx]->AddMult(phi_gp_view, group_rhs);
            }
            
            // Utiliser forme condensée ou mixte selon la configuration
            if (use_condensed_form_) {
                SolveGroupCondensed(g, group_rhs, solver_type);
            } else {
                SolveGroupInternal(g, group_rhs, solver_type);
            }
        }

        double prod_new = 0.0;
        for (int g = 0; g < ng; ++g) {
            phi_g_view.SetDataAndSize(Sol_Phi_gf_->GetData() + g * n_phi_dofs, n_phi_dofs);
            M_fiss_[g]->Mult(phi_g_view, prod_g);
            prod_new += prod_g.Sum();
        }
        
	double keff_new ;
	
        if (it > 2) {
	    keff_new = keff * (prod_new / prod_old) ;
	    diff_k = std::abs(keff_new - keff);
	}
	else {
            keff_new = keff;
	}

        *Sol_Phi_gf_ /= Sol_Phi_gf_->Norml2();

        if (it > 2) accel(*Sol_Phi_gf_);

        if (verbosity_ >= VerbosityLevel::NORMAL && (it % 5 == 0 || diff_k < params_.tol_keff)) {
            std::cout << "  It " << std::setw(4) << it << ": k=" << std::fixed 
                      << std::setprecision(8) << keff << " (dk=" << std::scientific 
                      << std::setprecision(2) << diff_k << ")\n";
        }
    
    keff = keff_new ;
    }

    if (diff_k <= params_.tol_keff) {
        Log(VerbosityLevel::NORMAL, "\n✓ Convergence: k-eff = ", keff);
    } else {
        Log(VerbosityLevel::LIGHT, "\n⚠ Non convergé (dk = ", diff_k, ")");
    }

    has_valid_keff_ = true;
    last_keff_direct_ = keff;
    return keff;
}
// ============================================================================
// RÉSOLUTION ADJOINTE COMPLÈTE
// ============================================================================

double NeutMFEM::SolveAdjoint(LinearSolverType solver_type,
                             bool normalize_to_direct,
                             bool use_direct_keff) {
    Log(VerbosityLevel::NORMAL, "\n=== CALCUL ADJOINT ===");

    if (use_direct_keff) {
        if (!has_valid_keff_) {
            throw std::runtime_error("Aucun k-eff valide. Résolvez d'abord le problème direct.");
        }
        Log(VerbosityLevel::NORMAL, "Mode accéléré: k-eff fixe = ", last_keff_direct_);
        SolveAdjointFixed(last_keff_direct_, solver_type, normalize_to_direct);
        return last_keff_direct_;
    }

    if (Sol_Phi_gf_->Norml2() < 1e-12) {
        throw std::runtime_error("Le flux direct doit être calculé avant l'adjoint");
    }

    const int ng = n_grps_;
    const int n_phi_dofs = fes_Phi_->GetVSize();
    double keff_adj = 1.0;
    double diff_k = 1.0;

    ChebyshevAccelerator accel(10, 0.9);
    mfem::Vector total_fiss_adj(n_phi_dofs), prod_g(n_phi_dofs);
    mfem::Vector phi_adj_g, phi_adj_gp;

    for (int it = 0; it < params_.max_outer_iter && diff_k > params_.tol_keff; ++it) {
        // Source adjointe: Σ_g χ(g) · Φ†(g)
        total_fiss_adj = 0.0;
        for (int g = 0; g < ng; ++g) {
            phi_adj_g.SetDataAndSize(Sol_Phi_adj_gf_->GetData() + g * n_phi_dofs, n_phi_dofs);
            for (int i = 0; i < n_phi_dofs; ++i)
                total_fiss_adj(i) += (*Chi_gf_)(g * n_phi_dofs + i) * phi_adj_g(i);
        }

        double prod_old = 0.0;
        for (int g = 0; g < ng; ++g) {
            phi_adj_g.SetDataAndSize(Sol_Phi_adj_gf_->GetData() + g * n_phi_dofs, n_phi_dofs);
            M_fiss_[g]->Mult(phi_adj_g, prod_g);
            prod_old += prod_g.Sum();
        }
        total_fiss_adj *= (1.0 / keff_adj);

        // Résolution par groupe (ordre inverse)
        for (int g = ng - 1; g >= 0; --g) {
            mfem::Vector group_rhs(n_phi_dofs);
            group_rhs = 0.0;

            M_fiss_[g]->Mult(total_fiss_adj, prod_g);
            group_rhs += prod_g;

            // Scattering adjoint (indices transposés)
            for (int gp = 0; gp < ng; ++gp) {
                if (g == gp) continue;
                const int idx = gp * ng + g;  // Transposé
                if (!M_scatter_[idx]) continue;
                phi_adj_gp.SetDataAndSize(Sol_Phi_adj_gf_->GetData() + gp * n_phi_dofs, n_phi_dofs);
                M_scatter_[idx]->AddMult(phi_adj_gp, group_rhs);
            }
            
            // Utiliser forme condensée ou mixte selon la configuration
            if (use_condensed_form_) {
                SolveAdjointGroupCondensed(g, group_rhs, solver_type);
            } else {
                SolveAdjointGroupInternal(g, group_rhs, solver_type);
            }
        }

        double prod_new = 0.0;
        for (int g = 0; g < ng; ++g) {
            phi_adj_g.SetDataAndSize(Sol_Phi_adj_gf_->GetData() + g * n_phi_dofs, n_phi_dofs);
            M_fiss_[g]->Mult(phi_adj_g, prod_g);
            prod_new += prod_g.Sum();
        }

        double keff_new = keff_adj * (prod_new / prod_old);
        diff_k = std::abs(keff_new - keff_adj);
        keff_adj = keff_new;

        // Normalisation
        if (normalize_to_direct) {
            double inner = 0.0;
            mfem::Vector phi_g;
            for (int g = 0; g < ng; ++g) {
                phi_g.SetDataAndSize(Sol_Phi_gf_->GetData() + g * n_phi_dofs, n_phi_dofs);
                phi_adj_g.SetDataAndSize(Sol_Phi_adj_gf_->GetData() + g * n_phi_dofs, n_phi_dofs);
                M_fiss_[g]->Mult(phi_g, prod_g);
                inner += (phi_adj_g * prod_g);
            }
            if (std::abs(inner) > 1e-14)
                *Sol_Phi_adj_gf_ *= (1.0 / inner);
        } else {
            *Sol_Phi_adj_gf_ /= Sol_Phi_adj_gf_->Norml2();
        }

        if (it > 2) accel(*Sol_Phi_adj_gf_);

        if (verbosity_ >= VerbosityLevel::NORMAL && (it % 10 == 0 || diff_k < params_.tol_keff)) {
            std::cout << "  It " << std::setw(4) << it << ": k†=" << std::fixed 
                      << std::setprecision(8) << keff_adj << " (dk=" << std::scientific 
                      << std::setprecision(2) << diff_k << ")\n";
        }
    }

    Log(VerbosityLevel::NORMAL, diff_k <= params_.tol_keff ? "✓ Convergence adjointe" : "⚠ Non convergé");
    return keff_adj;
}

void NeutMFEM::SolveAdjointFixed(double keff_known, LinearSolverType solver_type,
                                bool normalize_to_direct) {
    Log(VerbosityLevel::NORMAL, "Résolution adjointe à k-eff fixe = ", keff_known);

    if (keff_known <= 0.0) throw std::invalid_argument("k-eff doit être positif");

    const int ng = n_grps_;
    const int n_phi_dofs = fes_Phi_->GetVSize();

    mfem::Vector total_fiss_adj(n_phi_dofs), prod_g(n_phi_dofs);
    mfem::Vector phi_adj_g, phi_adj_gp, phi_adj_old(Sol_Phi_adj_gf_->Size());

    double diff_phi = 1.0;
    int iteration = 0;

    while (diff_phi > params_.tol_flux && iteration < params_.max_outer_iter) {
        phi_adj_old = *Sol_Phi_adj_gf_;

        total_fiss_adj = 0.0;
        for (int g = 0; g < ng; ++g) {
            phi_adj_g.SetDataAndSize(Sol_Phi_adj_gf_->GetData() + g * n_phi_dofs, n_phi_dofs);
            for (int i = 0; i < n_phi_dofs; ++i)
                total_fiss_adj(i) += (*Chi_gf_)(g * n_phi_dofs + i) * phi_adj_g(i);
        }
        total_fiss_adj *= (1.0 / keff_known);

        for (int g = ng - 1; g >= 0; --g) {
            mfem::Vector group_rhs(n_phi_dofs);
            group_rhs = 0.0;
            M_fiss_[g]->Mult(total_fiss_adj, prod_g);
            group_rhs += prod_g;

            for (int gp = 0; gp < ng; ++gp) {
                if (g == gp) continue;
                const int idx = gp * ng + g;
                if (!M_scatter_[idx]) continue;
                phi_adj_gp.SetDataAndSize(Sol_Phi_adj_gf_->GetData() + gp * n_phi_dofs, n_phi_dofs);
                M_scatter_[idx]->AddMult(phi_adj_gp, group_rhs);
            }
            
            // Utiliser forme condensée ou mixte selon la configuration
            if (use_condensed_form_) {
                SolveAdjointGroupCondensed(g, group_rhs, solver_type);
            } else {
                SolveAdjointGroupInternal(g, group_rhs, solver_type);
            }
        }

        if (normalize_to_direct) {
            double inner = 0.0;
            mfem::Vector phi_g;
            for (int g = 0; g < ng; ++g) {
                phi_g.SetDataAndSize(Sol_Phi_gf_->GetData() + g * n_phi_dofs, n_phi_dofs);
                phi_adj_g.SetDataAndSize(Sol_Phi_adj_gf_->GetData() + g * n_phi_dofs, n_phi_dofs);
                M_fiss_[g]->Mult(phi_g, prod_g);
                inner += (phi_adj_g * prod_g);
            }
            if (std::abs(inner) > 1e-14)
                *Sol_Phi_adj_gf_ *= (1.0 / inner);
        } else {
            *Sol_Phi_adj_gf_ /= Sol_Phi_adj_gf_->Norml2();
        }

        phi_adj_old -= *Sol_Phi_adj_gf_;
        diff_phi = phi_adj_old.Norml2() / Sol_Phi_adj_gf_->Norml2();
        ++iteration;

        if (verbosity_ >= VerbosityLevel::NORMAL && (iteration % 5 == 0 || diff_phi < params_.tol_flux)) {
            std::cout << "  It " << std::setw(3) << iteration << ": ||dΦ†||/||Φ†|| = " 
                      << std::scientific << std::setprecision(3) << diff_phi << "\n";
        }
    }
    Log(VerbosityLevel::NORMAL, "Adjoint fixe terminé en ", iteration, " itérations");
}

// ============================================================================
// IMPORTANCE ET PRODUCTION ADJOINTE
// ============================================================================

double NeutMFEM::GetGroupImportance(int group_idx) {
    if (group_idx < 0 || group_idx >= n_grps_)
        throw std::out_of_range("Indice de groupe invalide");

    const int n_phi_dofs = fes_Phi_->GetVSize();
    mfem::Vector phi_g, phi_adj_g, prod_g(n_phi_dofs);

    phi_g.SetDataAndSize(Sol_Phi_gf_->GetData() + group_idx * n_phi_dofs, n_phi_dofs);
    phi_adj_g.SetDataAndSize(Sol_Phi_adj_gf_->GetData() + group_idx * n_phi_dofs, n_phi_dofs);

    M_fiss_[group_idx]->Mult(phi_g, prod_g);
    return phi_adj_g * prod_g;
}

double NeutMFEM::ComputeAdjointProduction() {
    double total = 0.0;
    for (int g = 0; g < n_grps_; ++g)
        total += GetGroupImportance(g);
    return total;
}

// ============================================================================
// RÉSOLUTION SOURCE FIXE
// ============================================================================

void NeutMFEM::SolveFixedSource(LinearSolverType solver_type) {
    Log(VerbosityLevel::NORMAL, "\n=== RÉSOLUTION SOURCE FIXE ===");

    const int n_phi_dofs = fes_Phi_->GetVSize();

    for (int g = 0; g < n_grps_; ++g) {
        Log(VerbosityLevel::LIGHT, "  Groupe ", g + 1, "/", n_grps_);

        mfem::GridFunction src_g(fes_Mat_);
        src_g.SetData(SRC_gf_->GetData() + GetGroupOffset(g));

        mfem::GridFunctionCoefficient src_coeff(&src_g);
        mfem::LinearForm lf(fes_Phi_);
        lf.AddDomainIntegrator(new mfem::DomainLFIntegrator(src_coeff));
        lf.Assemble();

        mfem::Vector group_source(n_phi_dofs);
        group_source = lf;

        // Utiliser forme condensée ou mixte selon la configuration
        if (use_condensed_form_) {
            SolveGroupCondensed(g, group_source, solver_type);
        } else {
            SolveGroupInternal(g, group_source, solver_type);
        }
    }

    Log(VerbosityLevel::NORMAL, "✓ Source fixe terminée\n");
}

// ============================================================================
// RECONSTRUCTION DES COURANTS
// ============================================================================

void NeutMFEM::ReconstructAllCurrents() {
    Log(VerbosityLevel::NORMAL, "Reconstruction des courants...");
    
    for (int g = 0; g < n_grps_; ++g) {
        ReconstructCurrent(g);
    }
    
    Log(VerbosityLevel::NORMAL, "✓ Courants reconstruits");
}

// ============================================================================
// RÉSOLUTION SOUS-CRITIQUE
// ============================================================================

double NeutMFEM::SolveSubcritical(LinearSolverType solver_type) {
    Log(VerbosityLevel::NORMAL, "\n=== CALCUL SOUS-CRITIQUE ===");

    const int ng = n_grps_;
    const int n_phi_dofs = fes_Phi_->GetVSize();

    // Vérification source externe
    double ext_src_total = 0.0;
    for (int g = 0; g < ng; ++g) {
        mfem::GridFunction src_g(fes_Mat_);
        src_g.SetData(SRC_gf_->GetData() + GetGroupOffset(g));
        for (int i = 0; i < mesh_->GetNE(); ++i) {
            auto* T = mesh_->GetElementTransformation(i);
            const auto* ir = &mfem::IntRules.Get(mesh_->GetElementGeometry(i), 
                                                  2 * fes_Mat_->GetMaxElementOrder());
            for (int j = 0; j < ir->GetNPoints(); ++j) {
                const auto& ip = ir->IntPoint(j);
                T->SetIntPoint(&ip);
                ext_src_total += src_g.GetValue(*T, ip) * ip.weight * T->Weight();
            }
        }
    }

    if (ext_src_total < 1e-14)
        throw std::runtime_error("Source externe nulle");

    Log(VerbosityLevel::NORMAL, "Source externe totale: ", ext_src_total, " n/s");

    *Sol_Phi_gf_ = 1.0;

    double diff_flux = 1.0;
    int iteration = 0;

    ChebyshevAccelerator accel(10, 0.9);
    mfem::Vector total_fiss(n_phi_dofs), prod_g(n_phi_dofs);
    mfem::Vector phi_g_view, phi_gp_view, phi_old(Sol_Phi_gf_->Size());

    while (diff_flux > params_.tol_flux && iteration < params_.max_outer_iter) {
        phi_old = *Sol_Phi_gf_;

        total_fiss = 0.0;
        for (int g = 0; g < ng; ++g) {
            phi_g_view.SetDataAndSize(Sol_Phi_gf_->GetData() + g * n_phi_dofs, n_phi_dofs);
            M_fiss_[g]->AddMult(phi_g_view, total_fiss);
        }

        for (int g = 0; g < ng; ++g) {
            mfem::Vector group_rhs(n_phi_dofs);
            group_rhs = 0.0;

            // Source externe
            mfem::GridFunction src_g(fes_Mat_);
            src_g.SetData(SRC_gf_->GetData() + GetGroupOffset(g));
            mfem::GridFunctionCoefficient src_coeff(&src_g);
            mfem::LinearForm src_lf(fes_Phi_);
            src_lf.AddDomainIntegrator(new mfem::DomainLFIntegrator(src_coeff));
            src_lf.Assemble();
            group_rhs += src_lf;

            // Fission
            for (int i = 0; i < n_phi_dofs; ++i)
                group_rhs(i) += (*Chi_gf_)(g * n_phi_dofs + i) * total_fiss(i);

            // Scattering
            for (int gp = 0; gp < ng; ++gp) {
                if (g == gp) continue;
                const int idx = g * ng + gp;
                if (!M_scatter_[idx]) continue;
                phi_gp_view.SetDataAndSize(Sol_Phi_gf_->GetData() + gp * n_phi_dofs, n_phi_dofs);
                M_scatter_[idx]->AddMult(phi_gp_view, group_rhs);
            }

            // Utiliser forme condensée ou mixte selon la configuration
            if (use_condensed_form_) {
                SolveGroupCondensed(g, group_rhs, solver_type);
            } else {
                SolveGroupInternal(g, group_rhs, solver_type);
            }
        }

        mfem::Vector diff_vec(Sol_Phi_gf_->Size());
        add(*Sol_Phi_gf_, -1.0, phi_old, diff_vec);
        diff_flux = diff_vec.Norml2() / Sol_Phi_gf_->Norml2();

        if (iteration > 2) accel(*Sol_Phi_gf_);
        ++iteration;

        if (verbosity_ >= VerbosityLevel::NORMAL && (iteration % 5 == 0 || diff_flux < params_.tol_flux)) {
            std::cout << "  It " << std::setw(4) << iteration << ": ||dφ||/||φ|| = " 
                      << std::scientific << std::setprecision(3) << diff_flux << "\n";
        }
    }

    // Calcul M = Production / Source
    double fiss_prod = 0.0;
    for (int g = 0; g < ng; ++g) {
        phi_g_view.SetDataAndSize(Sol_Phi_gf_->GetData() + g * n_phi_dofs, n_phi_dofs);
        M_fiss_[g]->Mult(phi_g_view, prod_g);
        fiss_prod += prod_g.Sum();
    }

    double M = fiss_prod / ext_src_total;
    Log(VerbosityLevel::NORMAL, "✓ Facteur de multiplication M = ", M);
    Log(VerbosityLevel::NORMAL, "  (k_eff ≈ ", 1.0 - 1.0/M, ")\n");

    return M;
}

// ============================================================================
// EXPORT VTK
// ============================================================================

void NeutMFEM::SaveVTK(std::string filename_prefix, bool coarse, bool zoom) {
    Log(VerbosityLevel::NORMAL, "Export VTK: ", filename_prefix);

    mfem::ParaViewDataCollection pd(filename_prefix, mesh_);
    pd.SetPrefixPath("ParaView");
    pd.SetDataFormat(mfem::VTKFormat::BINARY);

    // Vecteurs temporaires pour éviter les dangling pointers
    std::vector<mfem::GridFunction*> temp_gfs;

    for (int g = 0; g < n_grps_; ++g) {
        std::string suffix = "_G" + std::to_string(g + 1);
        const int offset_phi = GetGroupOffsetPhi(g);
        const int offset_mat = GetGroupOffset(g);

        auto* tmp_phi = new mfem::GridFunction(fes_Phi_);
        tmp_phi->SetData(Sol_Phi_gf_->GetData() + offset_phi);
        pd.RegisterField("Flux" + suffix, tmp_phi);
        temp_gfs.push_back(tmp_phi);

        auto* tmp_adj = new mfem::GridFunction(fes_Phi_);
        tmp_adj->SetData(Sol_Phi_adj_gf_->GetData() + offset_phi);
        pd.RegisterField("FluxAdj" + suffix, tmp_adj);
        temp_gfs.push_back(tmp_adj);

        auto* tmp_d = new mfem::GridFunction(fes_Mat_);
        tmp_d->SetData(D_gf_->GetData() + offset_mat);
        pd.RegisterField("D" + suffix, tmp_d);
        temp_gfs.push_back(tmp_d);

        auto* tmp_sigr = new mfem::GridFunction(fes_Mat_);
        tmp_sigr->SetData(SigR_gf_->GetData() + offset_mat);
        pd.RegisterField("SigR" + suffix, tmp_sigr);
        temp_gfs.push_back(tmp_sigr);

        auto* tmp_nsf = new mfem::GridFunction(fes_Mat_);
        tmp_nsf->SetData(NSF_gf_->GetData() + offset_mat);
        pd.RegisterField("NSF" + suffix, tmp_nsf);
        temp_gfs.push_back(tmp_nsf);

        auto* tmp_chi = new mfem::GridFunction(fes_Mat_);
        tmp_chi->SetData(Chi_gf_->GetData() + offset_mat);
        pd.RegisterField("Chi" + suffix, tmp_chi);
        temp_gfs.push_back(tmp_chi);

        auto* tmp_src = new mfem::GridFunction(fes_Mat_);
        tmp_src->SetData(SRC_gf_->GetData() + offset_mat);
        pd.RegisterField("SRC" + suffix, tmp_src);
        temp_gfs.push_back(tmp_src);
    }

    pd.Save();
    
    // Nettoyage (les GridFunctions sont copiées par ParaView)
    for (auto* gf : temp_gfs) delete gf;
    
    Log(VerbosityLevel::NORMAL, "Export terminé\n");
    
    if (coarse) {
        SaveVTK_Coarse(filename_prefix + "_coarse") ; 
    }

    if (zoom) {
        SaveVTK_Zoom(filename_prefix + "_zoom") ; 
    }
}

void NeutMFEM::SaveVTK_Coarse(std::string filename_prefix) {
    if (!mesh_coarse_ || !Sol_Phi_coarse_gf_) {
        Log(VerbosityLevel::LIGHT, "⚠ Pas de données coarse disponibles pour l'export VTK");
        return;
    }

    Log(VerbosityLevel::NORMAL, "Export VTK Coarse: ", filename_prefix);

    mfem::ParaViewDataCollection pd(filename_prefix, mesh_coarse_);
    pd.SetPrefixPath("ParaView");
    pd.SetDataFormat(mfem::VTKFormat::BINARY);

    std::vector<mfem::GridFunction*> temp_gfs;
    const int n_mat_c = fes_Mat_coarse_->GetVSize();
    const int n_phi_c = fes_Phi_coarse_->GetVSize();

    for (int g = 0; g < n_grps_; ++g) {
        std::string suffix = "_G" + std::to_string(g + 1);

        // Flux coarse
        auto* tmp_phi = new mfem::GridFunction(fes_Phi_coarse_);
        tmp_phi->SetData(Sol_Phi_coarse_gf_->GetData() + g * n_phi_c);
        pd.RegisterField("Flux" + suffix, tmp_phi);
        temp_gfs.push_back(tmp_phi);

        // XS coarse
        if (D_coarse_gf_) {
            auto* tmp_d = new mfem::GridFunction(fes_Mat_coarse_);
            tmp_d->SetData(D_coarse_gf_->GetData() + g * n_mat_c);
            pd.RegisterField("D" + suffix, tmp_d);
            temp_gfs.push_back(tmp_d);
        }

        if (SigR_coarse_gf_) {
            auto* tmp_sigr = new mfem::GridFunction(fes_Mat_coarse_);
            tmp_sigr->SetData(SigR_coarse_gf_->GetData() + g * n_mat_c);
            pd.RegisterField("SigR" + suffix, tmp_sigr);
            temp_gfs.push_back(tmp_sigr);
        }

        if (NSF_coarse_gf_) {
            auto* tmp_nsf = new mfem::GridFunction(fes_Mat_coarse_);
            tmp_nsf->SetData(NSF_coarse_gf_->GetData() + g * n_mat_c);
            pd.RegisterField("NSF" + suffix, tmp_nsf);
            temp_gfs.push_back(tmp_nsf);
        }

        if (Chi_coarse_gf_) {
            auto* tmp_chi = new mfem::GridFunction(fes_Mat_coarse_);
            tmp_chi->SetData(Chi_coarse_gf_->GetData() + g * n_mat_c);
            pd.RegisterField("Chi" + suffix, tmp_chi);
            temp_gfs.push_back(tmp_chi);
        }

        if (SRC_coarse_gf_) {
            auto* tmp_src = new mfem::GridFunction(fes_Mat_coarse_);
            tmp_src->SetData(SRC_coarse_gf_->GetData() + g * n_mat_c);
            pd.RegisterField("SRC" + suffix, tmp_src);
            temp_gfs.push_back(tmp_src);
        }
    }

    pd.Save();
    for (auto* gf : temp_gfs) delete gf;

    Log(VerbosityLevel::NORMAL, "Export coarse terminé\n");
}

void NeutMFEM::SaveVTK_Zoom(std::string filename_prefix) {
    if (!Sol_Phi_zoom_gf_ || !mesh_zoom_) {
        Log(VerbosityLevel::LIGHT, "⚠ Pas de données zoom disponibles. Appelez zoom() d'abord.");
        return;
    }
    SaveVTK_Zoom(filename_prefix, Sol_Phi_zoom_gf_, zoom_factors_);
}

void NeutMFEM::SaveVTK_Zoom(std::string filename_prefix, 
                           mfem::GridFunction* phi_zoom,
                           const std::vector<int>& refine_factors) {
    if (!phi_zoom || !mesh_zoom_) {
        Log(VerbosityLevel::LIGHT, "⚠ Pas de données zoom disponibles pour l'export VTK");
        return;
    }

    Log(VerbosityLevel::NORMAL, "Export VTK Zoom: ", filename_prefix);

    // Créer un espace FE single-group pour l'export
    const int dim = mesh_zoom_->Dimension();
    mfem::L2_FECollection fec_export(0, dim);
    mfem::FiniteElementSpace fes_export(mesh_zoom_, &fec_export);
    
    const int n_phi_z = fes_export.GetVSize();

    mfem::ParaViewDataCollection pd(filename_prefix, mesh_zoom_);
    pd.SetPrefixPath("ParaView");
    pd.SetDataFormat(mfem::VTKFormat::BINARY);

    std::vector<mfem::GridFunction*> temp_gfs;

    for (int g = 0; g < n_grps_; ++g) {
        std::string suffix = "_G" + std::to_string(g + 1);

        // Créer une GridFunction single-group et copier les données
        auto* tmp_phi = new mfem::GridFunction(&fes_export);
        for (int i = 0; i < n_phi_z; ++i) {
            (*tmp_phi)(i) = (*phi_zoom)(g * n_phi_z + i);
        }
        pd.RegisterField("FluxZoom" + suffix, tmp_phi);
        temp_gfs.push_back(tmp_phi);
    }

    pd.Save();
    for (auto* gf : temp_gfs) delete gf;

    Log(VerbosityLevel::NORMAL, "Export zoom terminé\n");
}

// ============================================================================
// INTERPOLATION VERS MAILLAGE RAFFINÉ
// ============================================================================

void NeutMFEM::InterpolateConstantToRefinedMesh(const mfem::GridFunction& coarse_gf,
                                               mfem::GridFunction& fine_gf,
                                               const std::vector<int>& factors) const {
    const int dim = mesh_->Dimension();
    const int rx = factors[0];
    const int ry = (dim >= 2) ? factors[1] : 1;
    const int rz = (dim == 3) ? factors[2] : 1;

    // On récupère l'espace de la fonction fine pour manipuler les DOFs
    const mfem::FiniteElementSpace* fine_fes = fine_gf.FESpace();
    const mfem::FiniteElementSpace* coarse_fes = coarse_gf.FESpace();

    for (int iz = 0; iz < nz_; ++iz) {
        for (int iy = 0; iy < ny_; ++iy) {
            for (int ix = 0; ix < nx_; ++ix) {
                
                // 1. Récupérer la valeur de l'élément grossier (sécurisé)
                const int coarse_el_idx = ix + iy * nx_ + iz * nx_ * ny_;
                mfem::Array<int> c_vdofs;
                coarse_fes->GetElementVDofs(coarse_el_idx, c_vdofs);
                const double value = coarse_gf(c_vdofs[0]); // On prend la valeur constante

                // 2. Remplir les éléments fins correspondants
                for (int kz = 0; kz < rz; ++kz) {
                    for (int ky = 0; ky < ry; ++ky) {
                        for (int kx = 0; kx < rx; ++kx) {
                            const int f_ix = ix * rx + kx;
                            const int f_iy = iy * ry + ky;
                            const int f_iz = iz * rz + kz;
                            
                            const int fine_el_idx = f_ix + f_iy * (nx_ * rx) + 
                                                   f_iz * (nx_ * rx) * (ny_ * ry);

                            // CORRECTION : Appliquer la valeur à TOUS les DOFs de l'élément fin
                            mfem::Array<int> f_vdofs;
                            fine_fes->GetElementVDofs(fine_el_idx, f_vdofs);
                            for (int i = 0; i < f_vdofs.Size(); ++i) {
                                fine_gf(f_vdofs[i]) = value;
                            }
                        }
                    }
                }
            }
        }
    }
}
// ============================================================================
// ZOOM (RAFFINEMENT) - Cohérent avec SolveKeff
// ============================================================================

mfem::GridFunction* NeutMFEM::zoom(std::vector<int> refine, LinearSolverType solver_type) {
    if (refine.size() < 3) refine.resize(3, 1);
    if (refine[0] <= 0 || refine[1] <= 0 || refine[2] <= 0)
        throw std::invalid_argument("Facteurs de raffinement doivent être positifs");

    Log(VerbosityLevel::NORMAL, "\n=== ZOOM (RAFFINEMENT) ===");
    
    const int dim = mesh_->Dimension();
    const int ng = n_grps_;
    const int rx = refine[0];
    const int ry = (dim >= 2) ? refine[1] : 1;
    const int rz = (dim == 3) ? refine[2] : 1;

    Log(VerbosityLevel::NORMAL, "Facteurs: ", rx, " × ", ry, " × ", rz);

    // ========================================================================
    // 1. CRÉATION DU MAILLAGE FIN
    // ========================================================================
    std::set<double> x_set, y_set, z_set;
    for (int i = 0; i < mesh_->GetNV(); ++i) {
        double* v = mesh_->GetVertex(i);
        x_set.insert(v[0]);
        if (dim >= 2) y_set.insert(v[1]);
        if (dim == 3) z_set.insert(v[2]);
    }

    std::vector<double> x_coarse(x_set.begin(), x_set.end());
    std::vector<double> y_coarse = (dim >= 2) ? std::vector<double>(y_set.begin(), y_set.end())
                                              : std::vector<double>{0.0, 1.0};
    std::vector<double> z_coarse = (dim == 3) ? std::vector<double>(z_set.begin(), z_set.end())
                                              : std::vector<double>{0.0};

    std::vector<double> x_fine, y_fine, z_fine;
    for (size_t i = 0; i + 1 < x_coarse.size(); ++i)
        for (int k = 0; k < rx; ++k)
            x_fine.push_back(x_coarse[i] + k * (x_coarse[i+1] - x_coarse[i]) / rx);
    x_fine.push_back(x_coarse.back());

    if (dim >= 2) {
        for (size_t i = 0; i + 1 < y_coarse.size(); ++i)
            for (int k = 0; k < ry; ++k)
                y_fine.push_back(y_coarse[i] + k * (y_coarse[i+1] - y_coarse[i]) / ry);
        y_fine.push_back(y_coarse.back());
    } else y_fine = {0.0, 1.0};

    if (dim == 3) {
        for (size_t i = 0; i + 1 < z_coarse.size(); ++i)
            for (int k = 0; k < rz; ++k)
                z_fine.push_back(z_coarse[i] + k * (z_coarse[i+1] - z_coarse[i]) / rz);
        z_fine.push_back(z_coarse.back());
    } else z_fine = {0.0};

    const int nx_fine = x_fine.size() - 1;
    const int ny_fine = (dim >= 2) ? y_fine.size() - 1 : 1;
    const int nz_fine = (dim == 3) ? z_fine.size() - 1 : 1;

    Log(VerbosityLevel::NORMAL, "Maillage: ", nx_, "×", ny_, "×", nz_, 
        " → ", nx_fine, "×", ny_fine, "×", nz_fine);

    mfem::Mesh* mesh_fine;
    if (dim == 3)
        mesh_fine = new mfem::Mesh(mfem::Mesh::MakeCartesian3D(
            nx_fine, ny_fine, nz_fine, mfem::Element::HEXAHEDRON, 1.0, 1.0, 1.0, false));
    else if (dim == 2)
        mesh_fine = new mfem::Mesh(mfem::Mesh::MakeCartesian2D(
            nx_fine, ny_fine, mfem::Element::QUADRILATERAL, false, 1.0, 1.0, false));
    else
        mesh_fine = new mfem::Mesh(mfem::Mesh::MakeCartesian1D(nx_fine, 1.0));

    for (int i = 0; i < mesh_fine->GetNV(); ++i) {
        double* v = mesh_fine->GetVertex(i);
        v[0] = x_fine[std::min(static_cast<int>(std::round(v[0] * nx_fine)), nx_fine)];
        if (dim >= 2) v[1] = y_fine[std::min(static_cast<int>(std::round(v[1] * ny_fine)), ny_fine)];
        if (dim == 3) v[2] = z_fine[std::min(static_cast<int>(std::round(v[2] * nz_fine)), nz_fine)];
    }
    mesh_fine->Finalize(true);

    // ========================================================================
    // 2. ESPACES EF FINS
    // ========================================================================
    auto* fec_Phi_fine = new mfem::L2_FECollection(fec_Phi_->GetOrder(), dim);
    auto* fec_J_fine = new mfem::RT_FECollection(fec_J_->GetOrder(), dim);

    // Espace simple groupe pour les calculs intermédiaires
    auto* fes_Phi_fine_sg = new mfem::FiniteElementSpace(mesh_fine, fec_Phi_fine);
    auto* fes_J_fine_sg = new mfem::FiniteElementSpace(mesh_fine, fec_J_fine);

    // Espace multi-groupes : Utilisation impérative de byVDIM pour votre indexation
    auto* fes_Phi_fine_mg = new mfem::FiniteElementSpace(
        mesh_fine, fec_Phi_fine, ng, mfem::Ordering::byVDIM);
    auto* fes_J_fine_mg = new mfem::FiniteElementSpace(
        mesh_fine, fec_J_fine, ng, mfem::Ordering::byVDIM);

    const int nPhi_fine = fes_Phi_fine_sg->GetVSize();
    const int nJ_fine = fes_J_fine_sg->GetVSize();
    const int n_coarse_elems = mesh_->GetNE();

    // Lambda pour interpolation coarse → fine
    auto coarse_to_fine_idx = [&](int i_fine) -> int {
        int ix_f = i_fine % nx_fine;
        int iy_f = (dim >= 2) ? (i_fine / nx_fine) % ny_fine : 0;
        int iz_f = (dim == 3) ? i_fine / (nx_fine * ny_fine) : 0;
        return (ix_f / rx) + (iy_f / ry) * nx_ + (iz_f / rz) * nx_ * ny_;
    };

    // ========================================================================
    // 4. RÉSOLUTION GROUPE PAR GROUPE
    // ========================================================================
    auto* Sol_Phi_fine = new mfem::GridFunction(fes_Phi_fine_mg);
    *Sol_Phi_fine = 0.0;

    // Calcul de la source de fission globale (basée sur la solution grossière convergée)
    double keff_inv = 1.0 / (has_valid_keff_ ? last_keff_direct_ : 1.0);
    mfem::Vector total_fiss_fine(nPhi_fine);
    total_fiss_fine = 0.0;

    for (int gp = 0; gp < ng; ++gp) {
        // Vue sur le groupe gp du NSF grossier
        mfem::GridFunction nsf_coarse_view(fes_Phi_);
        nsf_coarse_view.SetData(NSF_gf_->GetData() + gp * n_coarse_elems);
        
        // Vue sur le groupe gp du Flux grossier
        mfem::GridFunction phi_coarse_view(fes_Phi_);
        phi_coarse_view.SetData(Sol_Phi_gf_->GetData() + gp * n_coarse_elems);

        mfem::GridFunction nsf_fine(fes_Phi_fine_sg);
        mfem::GridFunction phi_fine(fes_Phi_fine_sg);

        InterpolateConstantToRefinedMesh(nsf_coarse_view, nsf_fine, refine);
        InterpolateConstantToRefinedMesh(phi_coarse_view, phi_fine, refine);

        for (int i = 0; i < nPhi_fine; ++i) {
            total_fiss_fine(i) += nsf_fine(i) * phi_fine(i) * keff_inv;
        }
    }

    // Après avoir interpolé d_f et sigr_f :
    auto filename_prefix = "DebugZoom" ;
    mfem::ParaViewDataCollection pd(filename_prefix, mesh_fine);  // Utiliser mesh_fine, pas mesh_zoom_
    pd.SetPrefixPath("ParaView");
    pd.SetDataFormat(mfem::VTKFormat::BINARY);


    // 4. RÉSOLUTION GROUPE PAR GROUPE
    for (int g = 0; g < ng; ++g) {
        Log(VerbosityLevel::LIGHT, "  Groupe ", g + 1, "/", ng);

        // --- Préparation des paramètres physiques fins pour ce groupe ---
        mfem::GridFunction d_coarse_v(fes_Phi_), sigr_coarse_v(fes_Phi_), chi_coarse_v(fes_Phi_);
        d_coarse_v.SetData(D_gf_->GetData() + g * n_coarse_elems);
        sigr_coarse_v.SetData(SigR_gf_->GetData() + g * n_coarse_elems);
        chi_coarse_v.SetData(Chi_gf_->GetData() + g * n_coarse_elems);

        mfem::GridFunction d_f(fes_Phi_fine_sg), sigr_f(fes_Phi_fine_sg), chi_f(fes_Phi_fine_sg);
        InterpolateConstantToRefinedMesh(d_coarse_v, d_f, refine);
        InterpolateConstantToRefinedMesh(sigr_coarse_v, sigr_f, refine);
        InterpolateConstantToRefinedMesh(chi_coarse_v, chi_f, refine);
	
        std::string suffix = "_G" + std::to_string(g + 1);
	auto* tmp_src = new mfem::GridFunction(fes_Phi_fine_sg);
	tmp_src->SetData(d_f.GetData());
	pd.RegisterField("D" + suffix, tmp_src);
	
        // --- Construction du RHS (Fission) ---
        mfem::Vector group_rhs(nPhi_fine);
        for (int i = 0; i < nPhi_fine; ++i) group_rhs(i) = chi_f(i) * total_fiss_fine(i);

        // --- Assemblage des matrices fines ---
        InverseCoefficient invD(&d_f);
        mfem::BilinearForm a_form(fes_J_fine_sg);
        a_form.AddDomainIntegrator(new mfem::VectorFEMassIntegrator(invD));
        a_form.Assemble(); a_form.Finalize();

        mfem::GridFunctionCoefficient sigR_coeff(&sigr_f);
        mfem::BilinearForm c_form(fes_Phi_fine_sg);
        c_form.AddDomainIntegrator(new mfem::MassIntegrator(sigR_coeff));
        c_form.Assemble(); c_form.Finalize();

        mfem::MixedBilinearForm b_form(fes_J_fine_sg, fes_Phi_fine_sg);
        b_form.AddDomainIntegrator(new mfem::VectorFEDivergenceIntegrator);
        b_form.Assemble(); b_form.Finalize();

        mfem::SparseMatrix *BT_fine = mfem::Transpose(b_form.SpMat());

        // --- Solveur Bloc ---
        mfem::Array<int> offsets(3);
        offsets[0] = 0; offsets[1] = nJ_fine; offsets[2] = nPhi_fine;
        offsets.PartialSum();

        mfem::BlockVector x_block(offsets), b_block(offsets);
        b_block = 0.0;
        b_block.GetBlock(1) = group_rhs;

        mfem::BlockOperator op(offsets);
        op.SetBlock(0, 0, &a_form.SpMat());
        op.SetBlock(0, 1, BT_fine);
        op.SetBlock(1, 0, &b_form.SpMat(), -1.0);
        op.SetBlock(1, 1, &c_form.SpMat());

        std::unique_ptr<mfem::IterativeSolver> solver(CreateLinearSolver(solver_type));
        solver->SetOperator(op);
        solver->Mult(b_block, x_block);

        // Transfert vers la solution multi-groupe
        for (int i = 0; i < nPhi_fine; ++i) (*Sol_Phi_fine)(g * nPhi_fine + i) = x_block.GetBlock(1)(i);

        delete BT_fine;
    }

    Log(VerbosityLevel::NORMAL, " Fichier debug exporté : DebugZoom/DebugZoom_000000.pvtu");
    pd.Save();

    // 5. MISE À JOUR DE L'ÉTAT ET NETTOYAGE
    delete Sol_Phi_zoom_gf_;
    delete fes_Phi_zoom_;
    delete mesh_zoom_;

    mesh_zoom_ = mesh_fine;
    fes_Phi_zoom_ = fes_Phi_fine_mg;
    Sol_Phi_zoom_gf_ = Sol_Phi_fine;

    // Suppression des espaces sg temporaires
    delete fes_Phi_fine_sg; delete fes_J_fine_sg;
    // On ne supprime PAS fec_Phi_fine car fes_Phi_fine_mg en a besoin
    
    return Sol_Phi_zoom_gf_;
}

py::array_t<double> NeutMFEM::py_zoom(py::tuple pyremesh, bool adjoint,
                                     LinearSolverType solver_type) {
    auto subdiv = pyremesh.cast<std::vector<int>>();
    mfem::GridFunction* phi = zoom(subdiv, solver_type);
    if (!phi) return py::array_t<double>();
    
    // Dimensions du maillage zoomé
    const int dim = mesh_->Dimension();
    const int ng = n_grps_;
    const int rx = subdiv[0];
    const int ry = (subdiv.size() > 1) ? subdiv[1] : 1;
    const int rz = (subdiv.size() > 2) ? subdiv[2] : 1;
    
    const ssize_t nx_f = nx_ * rx;
    const ssize_t ny_f = (dim >= 2) ? ny_ * ry : 1;
    const ssize_t nz_f = (dim == 3) ? nz_ * rz : 1;
    const ssize_t n_elem_f = nx_f * ny_f * nz_f;
    
    // Utiliser les strides pour le flip Y sans copie (comme GetCoefArray)
    double* data = phi->GetData();
    const size_t d_size = sizeof(double);
    
    if (dim == 1) {
        return py::array_t<double>({(ssize_t)ng, nx_f},
            {(ssize_t)(nx_f * d_size), (ssize_t)d_size}, 
            data, py::cast(this));
    }
    if (dim == 2) {
        // Pointer vers la dernière ligne (haut de l'image)
        double* top_left = data + (ny_f - 1) * nx_f;
        return py::array_t<double>({(ssize_t)ng, ny_f, nx_f},
            {(ssize_t)(nx_f * ny_f * d_size), -(ssize_t)(nx_f * d_size), (ssize_t)d_size},
            top_left, py::cast(this));
    }
    // dim == 3
    double* start = data + (ny_f - 1) * nx_f;
    return py::array_t<double>({(ssize_t)ng, nz_f, ny_f, nx_f},
        {(ssize_t)(n_elem_f * d_size), (ssize_t)(nx_f * ny_f * d_size), 
         -(ssize_t)(nx_f * d_size), (ssize_t)d_size}, 
        start, py::cast(this));
}

// ============================================================================
// RÉFLECTEURS
// ============================================================================

int NeutMFEM::add_refl(py::array_t<double> D_arr, py::array_t<double> SigR_arr,
                      py::array_t<double> SigS_arr) {
    if (D_arr.ndim() != 1 || SigR_arr.ndim() != 1 || SigS_arr.ndim() != 2)
        throw std::invalid_argument("Dimensions incorrectes pour XS");

    if (D_arr.shape(0) != n_grps_ || SigR_arr.shape(0) != n_grps_ ||
        SigS_arr.shape(0) != n_grps_ || SigS_arr.shape(1) != n_grps_)
        throw std::invalid_argument("Tailles XS incompatibles");

    auto D_buf = D_arr.unchecked<1>();
    auto SigR_buf = SigR_arr.unchecked<1>();
    auto SigS_buf = SigS_arr.unchecked<2>();

    ReflectorCoefficients refl(n_grps_);
    mfem::Vector L(n_grps_);

    for (int g = 0; g < n_grps_; ++g) {
        double D = D_buf(g);
        double SigR = SigR_buf(g);
        double scatt_out = 0.0;
        for (int gp = 0; gp < n_grps_; ++gp) scatt_out += SigS_buf(g, gp);
        double SigA = std::max(SigR - scatt_out, 1e-10);
        L(g) = std::sqrt(D / SigA);
        refl.alpha(g) = D / L(g);
    }

    for (int g = 0; g < n_grps_; ++g) {
        for (int gp = 0; gp < g; ++gp) {
            double SigS_gp_g = SigS_buf(gp, g);
            if (std::abs(SigS_gp_g) > 1e-14) {
                refl.beta(gp, g) = -SigS_gp_g / (1.0/L(gp) + 1.0/L(g));
            }
        }
    }

    reflector_types_.push_back(refl);
    int id = static_cast<int>(reflector_types_.size()) - 1;
    Log(VerbosityLevel::NORMAL, "Réflecteur #", id, " créé");
    return id;
}

void NeutMFEM::set_refl(int refl_id, int dimension, bool is_upper) {
    if (refl_id < 0 || refl_id >= static_cast<int>(reflector_types_.size()))
        throw std::out_of_range("ID réflecteur invalide");

    const int mesh_dim = mesh_->Dimension();
    if (dimension < 0 || dimension >= mesh_dim)
        throw std::out_of_range("Dimension invalide");

    int bdr_attr = GetBoundaryAttribute(mesh_dim, dimension, is_upper);
    const auto& refl = reflector_types_[refl_id];

    std::pair<int, int> key(dimension, is_upper ? 1 : 0);
    reflector_rhs_[key].SetSize(n_grps_);
    reflector_rhs_[key] = 0.0;
    active_reflectors_[key] = refl_id;

    for (int g = 0; g < n_grps_; ++g) {
        SetBC(bdr_attr, BCType::ROBIN, 0.0);
        SetRobinCoefficients(bdr_attr, refl.alpha(g), 0.0);
    }

    Log(VerbosityLevel::NORMAL, "Réflecteur #", refl_id, " activé sur dim=", dimension,
        (is_upper ? " (max)" : " (min)"));
}

void NeutMFEM::clean_refl() {
    reflector_types_.clear();
    active_reflectors_.clear();
    reflector_rhs_.clear();
    Log(VerbosityLevel::NORMAL, "Réflecteurs supprimés");
}

void NeutMFEM::update_refl(const mfem::GridFunction& phi_current, int group_idx) {
    if (active_reflectors_.empty()) return;

    const int n_phi = fes_Phi_->GetVSize();
    const int mesh_dim = mesh_->Dimension();
    mfem::GridFunction phi_gp(fes_Phi_);

    for (const auto& kv : active_reflectors_) {
        const auto& key = kv.first;
        int refl_id = kv.second;
        int dimension = key.first;
        bool is_upper = (key.second == 1);

        const auto& refl = reflector_types_[refl_id];
        mfem::Vector& rhs_g = reflector_rhs_[key];

        if (group_idx == 0) rhs_g(group_idx) = 0.0;

        for (int gp = 0; gp < group_idx; ++gp) {
            double beta = refl.beta(gp, group_idx);
            if (std::abs(beta) < 1e-14) continue;

            phi_gp.SetData(const_cast<double*>(phi_current.GetData()) + gp * n_phi);
            int attr = GetBoundaryAttribute(mesh_dim, dimension, is_upper);

            double bnd_int = 0.0;
            for (int i = 0; i < mesh_->GetNBE(); ++i) {
                if (mesh_->GetBdrAttribute(i) != attr) continue;
                auto* T = mesh_->GetBdrFaceTransformations(i);
                if (!T) continue;

                const auto* ir = &mfem::IntRules.Get(
                    mesh_->GetBdrElement(i)->GetGeometryType(),
                    2 * fes_Phi_->GetMaxElementOrder() + 2);

                for (int j = 0; j < ir->GetNPoints(); ++j) {
                    const auto& ip = ir->IntPoint(j);
                    T->SetAllIntPoints(&ip);
                    bnd_int += beta * phi_gp.GetValue(*T, ip) * ip.weight * T->Weight();
                }
            }
            rhs_g(group_idx) += bnd_int;
        }
    }
}

// ============================================================================
// NETTOYAGE DES DONNÉES
// ============================================================================

void NeutMFEM::ClearEigenmodeData() {
    for (auto* gf : eigenvectors_) {
        delete gf;
    }
    eigenvectors_.clear();
    eigenvalues_.clear();
    
    for (auto* gf : eigenvectors_adj_) {
        delete gf;
    }
    eigenvectors_adj_.clear();
    eigenvalues_adj_.clear();
}

// ============================================================================
// PRODUIT SCALAIRE F-PONDÉRÉ : <φ₁, F·φ₂>
// ============================================================================

double NeutMFEM::FissionInnerProduct(const mfem::GridFunction& phi1,
                                      const mfem::GridFunction& phi2) {
    const int n_phi = fes_Phi_->GetVSize();
    double result = 0.0;
    mfem::Vector F_phi2(n_phi);  // F·φ₂ pour un groupe
    
    for (int g = 0; g < n_grps_; ++g) {
        mfem::Vector phi1_g, phi2_g;
        phi1_g.SetDataAndSize(const_cast<double*>(phi1.GetData()) + g * n_phi, n_phi);
        phi2_g.SetDataAndSize(const_cast<double*>(phi2.GetData()) + g * n_phi, n_phi);
        
        // F_phi2 = M_fiss[g] · φ₂_g  (contribution du groupe g à la fission)
        M_fiss_[g]->Mult(phi2_g, F_phi2);
        
        // result += φ₁_g · F_phi2
        result += phi1_g * F_phi2;
    }
    
    return result;
}

// ============================================================================
// ORTHOGONALISATION (GRAM-SCHMIDT MODIFIÉ AVEC PRODUIT F-PONDÉRÉ)
// ============================================================================

void NeutMFEM::OrthogonalizeMode(mfem::GridFunction& phi,
                                  const std::vector<mfem::GridFunction*>& previous_modes) {
    if (previous_modes.empty()) return;
    
    const int total_size = phi.Size();
    
    // Gram-Schmidt modifié avec produit <·, F·>
    for (const auto* mode_j : previous_modes) {
        // Calcul du coefficient : c_j = <φ, F·φ_j> / <φ_j, F·φ_j>
        double num = FissionInnerProduct(phi, *mode_j);
        double den = FissionInnerProduct(*mode_j, *mode_j);
        
        if (std::abs(den) > 1e-14) {
            double c_j = num / den;
            
            // φ = φ - c_j · φ_j
            for (int i = 0; i < total_size; ++i) {
                phi(i) -= c_j * (*mode_j)(i);
            }
        }
    }
}

// ============================================================================
// DÉFLATION DE HOTELLING
// ============================================================================

void NeutMFEM::ApplyHotellingDeflation(
    mfem::Vector& source,
    const std::vector<mfem::GridFunction*>& previous_modes,
    const std::vector<double>& previous_eigenvalues) {
    
    if (previous_modes.empty()) return;
    
    const int n_phi = fes_Phi_->GetVSize();
    mfem::Vector F_mode(n_phi);
    
    // Pour chaque mode précédent, soustraire sa contribution
    for (size_t j = 0; j < previous_modes.size(); ++j) {
        const auto* mode_j = previous_modes[j];
        
        // Calcul de <source, F·φ_j>
        double proj = 0.0;
        for (int g = 0; g < n_grps_; ++g) {
            mfem::Vector src_g, mode_j_g;
            src_g.SetDataAndSize(source.GetData() + g * n_phi, n_phi);
            mode_j_g.SetDataAndSize(const_cast<double*>(mode_j->GetData()) + g * n_phi, n_phi);
            
            M_fiss_[g]->Mult(mode_j_g, F_mode);
            proj += src_g * F_mode;
        }
        
        // Calcul de <φ_j, F·φ_j>
        double norm_F = FissionInnerProduct(*mode_j, *mode_j);
        
        if (std::abs(norm_F) > 1e-14) {
            double coeff = eigensolver_params_.deflation_weight * proj / norm_F;
            
            // source = source - coeff · F · φ_j
            for (int g = 0; g < n_grps_; ++g) {
                mfem::Vector src_g, mode_j_g;
                src_g.SetDataAndSize(source.GetData() + g * n_phi, n_phi);
                mode_j_g.SetDataAndSize(const_cast<double*>(mode_j->GetData()) + g * n_phi, n_phi);
                
                M_fiss_[g]->Mult(mode_j_g, F_mode);
                src_g.Add(-coeff, F_mode);
            }
        }
    }
}

// ============================================================================
// CALCUL D'UN MODE PROPRE UNIQUE
// ============================================================================

EigenMode NeutMFEM::ComputeSingleEigenmode(
    int mode_index,
    LinearSolverType solver_type,
    const std::vector<mfem::GridFunction*>& previous_modes,
    const std::vector<double>& previous_eigenvalues) {
    
    EigenMode result;
    result.mode_index = mode_index;
    
    const int n_phi = fes_Phi_->GetVSize();
    const int total_phi_size = n_grps_ * n_phi;
    
    // Créer l'eigenvector pour ce mode
    mfem::GridFunction* phi_n = new mfem::GridFunction(fes_Phi_mg_);
    
    // ------------------------------------------------------------------------
    // INITIALISATION
    // ------------------------------------------------------------------------
    if (mode_index == 0) {
        // Mode fondamental : initialisation uniforme ou depuis Sol_Phi_gf_
        if (Sol_Phi_gf_->Norml2() > 1e-10) {
            *phi_n = *Sol_Phi_gf_;
        } else {
            *phi_n = 1.0;
        }
    } else {
        // Modes supérieurs : initialisation avec bruit
        std::mt19937 rng(mode_index * 12345 + 67890);
        std::normal_distribution<double> dist(0.0, 1.0);
        
        for (int i = 0; i < total_phi_size; ++i) {
            (*phi_n)(i) = 1.0 + 0.3 * dist(rng);
        }
        
        // Orthogonaliser par rapport aux modes précédents
        OrthogonalizeMode(*phi_n, previous_modes);
    }
    
    // Normaliser
    double norm = phi_n->Norml2();
    if (norm > 1e-14) {
        *phi_n /= norm;
    }
    
    // ------------------------------------------------------------------------
    // ITÉRATION DE PUISSANCE INVERSE
    // ------------------------------------------------------------------------
    // On résout : L·φ^{new} = (1/λ_old) · F·φ^{old}
    // Ce qui revient à itérer sur L⁻¹·F avec estimation de 1/λ = k
    
    // Estimation initiale de k = 1/λ
    double k_est = (mode_index == 0) ? 1.0 : 
                   (previous_eigenvalues.empty() ? 1.0 : 
                    0.95 / previous_eigenvalues.back());  // k_n < k_{n-1}
    
    mfem::Vector fission_source(n_phi);
    mfem::Vector group_rhs(n_phi);
    mfem::Vector prod_g(n_phi);
    mfem::Vector phi_g_view, phi_gp_view;
    
    double diff_k = 1.0;
    int iter = 0;
    
    for (iter = 0; iter < eigensolver_params_.max_iter_per_mode; ++iter) {
        
        // ====================================================================
        // ÉTAPE 1 : Calculer la source de fission totale S = F·φ
        // ====================================================================
        mfem::Vector total_fission(total_phi_size);
        total_fission = 0.0;
        
        // total_fission_g = Σ_{g'} χ_g · νΣf_{g'} · φ_{g'}
        mfem::Vector fiss_sum(n_phi);
        fiss_sum = 0.0;
        
        for (int gp = 0; gp < n_grps_; ++gp) {
            phi_gp_view.SetDataAndSize(phi_n->GetData() + gp * n_phi, n_phi);
            M_fiss_[gp]->AddMult(phi_gp_view, fiss_sum);
        }
        
        double production = fiss_sum.Sum();
        
        // Distribuer avec le spectre χ et diviser par k (= 1/λ)
        for (int g = 0; g < n_grps_; ++g) {
            for (int i = 0; i < n_phi; ++i) {
                total_fission(g * n_phi + i) = 
                    (*Chi_gf_)(g * n_phi + i) * fiss_sum(i) / k_est;
            }
        }
        
        // ====================================================================
        // ÉTAPE 2 : Appliquer la déflation (pour modes n > 0)
        // ====================================================================
        if (mode_index > 0 && !previous_modes.empty()) {
            ApplyHotellingDeflation(total_fission, previous_modes, previous_eigenvalues);
        }
        
        // ====================================================================
        // ÉTAPE 3 : Résoudre L·φ^{new} = S groupe par groupe
        // ====================================================================
        for (int g = 0; g < n_grps_; ++g) {
            group_rhs = 0.0;
            
            // Source de fission pour ce groupe
            for (int i = 0; i < n_phi; ++i) {
                group_rhs(i) = total_fission(g * n_phi + i);
            }
            
            // Ajouter le scattering depuis les autres groupes
            for (int gp = 0; gp < n_grps_; ++gp) {
                if (g == gp) continue;
                int idx = g * n_grps_ + gp;
                if (!M_scatter_[idx]) continue;
                phi_gp_view.SetDataAndSize(phi_n->GetData() + gp * n_phi, n_phi);
                M_scatter_[idx]->AddMult(phi_gp_view, group_rhs);
            }
            
            // Stocker temporairement dans Sol_Phi_gf_ pour la résolution
            mfem::Vector sol_g;
            sol_g.SetDataAndSize(Sol_Phi_gf_->GetData() + g * n_phi, n_phi);
            phi_g_view.SetDataAndSize(phi_n->GetData() + g * n_phi, n_phi);
            sol_g = phi_g_view;  // Initial guess
            
            // Résoudre L_g · φ_g = rhs_g
            if (use_condensed_form_) {
                SolveGroupCondensed(g, group_rhs, solver_type);
            } else {
                SolveGroupInternal(g, group_rhs, solver_type);
            }
            
            // Récupérer le résultat
            phi_g_view = sol_g;
        }
        
        // ====================================================================
        // ÉTAPE 4 : Orthogonaliser (pour modes n > 0)
        // ====================================================================
        if (mode_index > 0 && eigensolver_params_.orthogonalize) {
            OrthogonalizeMode(*phi_n, previous_modes);
        }
        
        // ====================================================================
        // ÉTAPE 5 : Calculer la nouvelle production et estimer k = 1/λ
        // ====================================================================
        fiss_sum = 0.0;
        for (int g = 0; g < n_grps_; ++g) {
            phi_g_view.SetDataAndSize(phi_n->GetData() + g * n_phi, n_phi);
            M_fiss_[g]->AddMult(phi_g_view, fiss_sum);
        }
        double production_new = fiss_sum.Sum();
        
        double k_new = k_est * (production_new / production);
        diff_k = std::abs(k_new - k_est) / std::abs(k_est);
        k_est = k_new;
        
        // ====================================================================
        // ÉTAPE 6 : Normaliser l'eigenvector
        // ====================================================================
        if (eigensolver_params_.normalize_eigenvectors) {
            norm = phi_n->Norml2();
            if (norm > 1e-14) {
                *phi_n /= norm;
            }
        }
        
        // Log
        if (verbosity_ >= VerbosityLevel::DEBUG || 
            (verbosity_ >= VerbosityLevel::NORMAL && iter % 20 == 0)) {
            double lambda_est = 1.0 / k_est;
            std::cout << "  Mode " << mode_index << ", It " << std::setw(4) << iter
                      << ": λ = " << std::fixed << std::setprecision(8) << lambda_est
                      << " (k = " << k_est << ")"
                      << " (Δk/k = " << std::scientific << std::setprecision(2) << diff_k << ")\n";
        }
        
        // Vérifier convergence
        if (diff_k < eigensolver_params_.tol_eigenvalue) {
            result.converged = true;
            break;
        }
    }
    
    // ------------------------------------------------------------------------
    // STOCKER LES RÉSULTATS
    // ------------------------------------------------------------------------
    double lambda_final = 1.0 / k_est;
    
    result.lambda = lambda_final;
    result.keff = k_est;
    result.iterations = iter;
    
    if (!previous_eigenvalues.empty()) {
        result.dominance_ratio = lambda_final / previous_eigenvalues[0];
    } else {
        result.dominance_ratio = 1.0;
    }
    
    // Sauvegarder l'eigenvector et l'eigenvalue
    eigenvectors_.push_back(phi_n);
    eigenvalues_.push_back(lambda_final);
    
    return result;
}

// ============================================================================
// RÉSOLUTION DE TOUS LES MODES PROPRES
// ============================================================================

std::vector<EigenMode> NeutMFEM::SolveEigenmodes(
    int num_modes,
    LinearSolverType solver_type,
    const EigensolverParameters& params) {
    
    Log(VerbosityLevel::NORMAL, "\n=== CALCUL DES MODES PROPRES ===");
    Log(VerbosityLevel::NORMAL, "Problème : L·φ = λ·F·φ");
    Log(VerbosityLevel::NORMAL, "Nombre de modes demandés : ", num_modes);
    
    eigensolver_params_ = params;
    ClearEigenmodeData();
    
    std::vector<EigenMode> results;
    results.reserve(num_modes);
    
    for (int n = 0; n < num_modes; ++n) {
        Log(VerbosityLevel::NORMAL, "\n--- Calcul du mode ", n, " ---");
        
        // Collecter les modes précédents
        std::vector<mfem::GridFunction*> prev_modes;
        std::vector<double> prev_lambdas;
        for (int j = 0; j < n; ++j) {
            prev_modes.push_back(eigenvectors_[j]);
            prev_lambdas.push_back(eigenvalues_[j]);
        }
        
        // Calculer le mode n
        EigenMode mode = ComputeSingleEigenmode(n, solver_type, prev_modes, prev_lambdas);
        results.push_back(mode);
        
        // Log
        if (mode.converged) {
            Log(VerbosityLevel::NORMAL, "✓ Mode ", n, " convergé :");
            Log(VerbosityLevel::NORMAL, "    λ_", n, " = ", mode.lambda);
            Log(VerbosityLevel::NORMAL, "    k_", n, " = ", mode.keff, " (= 1/λ)");
            Log(VerbosityLevel::NORMAL, "    Itérations : ", mode.iterations);
        } else {
            Log(VerbosityLevel::LIGHT, "⚠ Mode ", n, " non convergé après ", 
                mode.iterations, " itérations");
        }
        
        if (n > 0) {
            Log(VerbosityLevel::NORMAL, "    Ratio λ_", n, "/λ_0 = ", mode.dominance_ratio);
        }
    }
    
    // Mettre à jour Sol_Phi_gf_ avec le mode fondamental
    if (!eigenvectors_.empty()) {
        *Sol_Phi_gf_ = *eigenvectors_[0];
        last_keff_direct_ = 1.0 / eigenvalues_[0];
        has_valid_keff_ = true;
    }
    
    // Résumé final
    Log(VerbosityLevel::NORMAL, "\n=== RÉSUMÉ DES EIGENVALUES ===");
    std::cout << std::fixed << std::setprecision(8);
    for (size_t n = 0; n < eigenvalues_.size(); ++n) {
        std::cout << "  λ_" << n << " = " << eigenvalues_[n]
                  << "   (k_" << n << " = " << 1.0/eigenvalues_[n] << ")";
        if (n > 0) {
            std::cout << "   ratio = " << std::setprecision(5) 
                      << eigenvalues_[n] / eigenvalues_[0];
        }
        std::cout << "\n";
    }
    
    return results;
}

// ============================================================================
// VERSION AVEC INITIALISATION COARSE
// ============================================================================

std::vector<EigenMode> NeutMFEM::SolveEigenmodes(
    int num_modes,
    LinearSolverType solver_type,
    bool use_coarse_init,
    const CoarseFactors& coarse_factors,
    const EigensolverParameters& params) {
    
    if (use_coarse_init && (coarse_factors.x > 1 || coarse_factors.y > 1 || coarse_factors.z > 1)) {
        Log(VerbosityLevel::NORMAL, "Initialisation coarse pour le mode fondamental...");
        
        // Utiliser SolveKeff avec coarse pour initialiser
        double k0 = SolveKeff(solver_type, true, coarse_factors, 30, 1e-3);
        double lambda0 = 1.0 / k0;
        
        eigensolver_params_ = params;
        ClearEigenmodeData();
        
        // Le mode fondamental est déjà dans Sol_Phi_gf_
        EigenMode mode0;
        mode0.mode_index = 0;
        mode0.lambda = lambda0;
        mode0.keff = k0;
        mode0.dominance_ratio = 1.0;
        mode0.converged = true;
        
        mfem::GridFunction* phi0 = new mfem::GridFunction(fes_Phi_mg_);
        *phi0 = *Sol_Phi_gf_;
        eigenvectors_.push_back(phi0);
        eigenvalues_.push_back(lambda0);
        
        std::vector<EigenMode> results;
        results.push_back(mode0);
        
        // Calculer les modes restants
        for (int n = 1; n < num_modes; ++n) {
            Log(VerbosityLevel::NORMAL, "\n--- Calcul du mode ", n, " ---");
            
            std::vector<mfem::GridFunction*> prev_modes;
            std::vector<double> prev_lambdas;
            for (int j = 0; j < n; ++j) {
                prev_modes.push_back(eigenvectors_[j]);
                prev_lambdas.push_back(eigenvalues_[j]);
            }
            
            EigenMode mode = ComputeSingleEigenmode(n, solver_type, prev_modes, prev_lambdas);
            results.push_back(mode);
        }
        
        return results;
    }
    
    return SolveEigenmodes(num_modes, solver_type, params);
}

// ============================================================================
// CALCUL D'UN MODE ADJOINT
// ============================================================================

EigenMode NeutMFEM::ComputeSingleAdjointEigenmode(
    int mode_index,
    LinearSolverType solver_type,
    const std::vector<mfem::GridFunction*>& previous_modes,
    const std::vector<double>& previous_eigenvalues) {
    
    EigenMode result;
    result.mode_index = mode_index;
    
    const int n_phi = fes_Phi_->GetVSize();
    const int total_phi_size = n_grps_ * n_phi;
    
    mfem::GridFunction* phi_adj_n = new mfem::GridFunction(fes_Phi_mg_);
    
    // Initialisation
    if (mode_index == 0 && Sol_Phi_adj_gf_->Norml2() > 1e-10) {
        *phi_adj_n = *Sol_Phi_adj_gf_;
    } else {
        std::mt19937 rng(mode_index * 54321 + 98765);
        std::normal_distribution<double> dist(0.0, 1.0);
        for (int i = 0; i < total_phi_size; ++i) {
            (*phi_adj_n)(i) = 1.0 + 0.3 * dist(rng);
        }
        OrthogonalizeMode(*phi_adj_n, previous_modes);
    }
    
    double norm = phi_adj_n->Norml2();
    if (norm > 1e-14) *phi_adj_n /= norm;
    
    // Estimation initiale de k
    double k_est = (mode_index == 0) ? 
                   (has_valid_keff_ ? last_keff_direct_ : 1.0) :
                   (previous_eigenvalues.empty() ? 1.0 : 0.95 / previous_eigenvalues.back());
    
    mfem::Vector fiss_source(n_phi), group_rhs(n_phi), prod_g(n_phi);
    mfem::Vector phi_adj_g, phi_adj_gp;
    
    double diff_k = 1.0;
    int iter = 0;
    
    for (iter = 0; iter < eigensolver_params_.max_iter_per_mode; ++iter) {
        // Source adjointe : S† = F†·φ† = Σ_g νΣf_g · φ†_g (puis multiplier par χ)
        mfem::Vector chi_weighted_source(total_phi_size);
        chi_weighted_source = 0.0;
        
        // D'abord calculer Σ_g χ_g · φ†_g
        mfem::Vector chi_phi_sum(n_phi);
        chi_phi_sum = 0.0;
        for (int g = 0; g < n_grps_; ++g) {
            phi_adj_g.SetDataAndSize(phi_adj_n->GetData() + g * n_phi, n_phi);
            for (int i = 0; i < n_phi; ++i) {
                chi_phi_sum(i) += (*Chi_gf_)(g * n_phi + i) * phi_adj_g(i);
            }
        }
        
        double production = 0.0;
        for (int g = 0; g < n_grps_; ++g) {
            phi_adj_g.SetDataAndSize(phi_adj_n->GetData() + g * n_phi, n_phi);
            M_fiss_[g]->Mult(phi_adj_g, prod_g);
            production += prod_g.Sum();
        }
        
        // Résoudre par groupe (ordre inverse pour l'adjoint)
        for (int g = n_grps_ - 1; g >= 0; --g) {
            group_rhs = 0.0;
            
            // Source adjointe : νΣf_g · (Σ_{g'} χ_{g'} · φ†_{g'}) / k
            M_fiss_[g]->Mult(chi_phi_sum, prod_g);
            group_rhs.Add(1.0 / k_est, prod_g);
            
            // Scattering adjoint (indices transposés)
            for (int gp = 0; gp < n_grps_; ++gp) {
                if (g == gp) continue;
                const int idx = gp * n_grps_ + g;  // Transposé !
                if (!M_scatter_[idx]) continue;
                phi_adj_gp.SetDataAndSize(phi_adj_n->GetData() + gp * n_phi, n_phi);
                M_scatter_[idx]->AddMult(phi_adj_gp, group_rhs);
            }
            
            // Résoudre
            mfem::Vector sol_adj_g;
            sol_adj_g.SetDataAndSize(Sol_Phi_adj_gf_->GetData() + g * n_phi, n_phi);
            phi_adj_g.SetDataAndSize(phi_adj_n->GetData() + g * n_phi, n_phi);
            sol_adj_g = phi_adj_g;
            
            if (use_condensed_form_) {
                SolveAdjointGroupCondensed(g, group_rhs, solver_type);
            } else {
                SolveAdjointGroupInternal(g, group_rhs, solver_type);
            }
            
            phi_adj_g = sol_adj_g;
        }
        
        // Orthogonaliser
        if (mode_index > 0 && eigensolver_params_.orthogonalize) {
            OrthogonalizeMode(*phi_adj_n, previous_modes);
        }
        
        // Estimer k
        double production_new = 0.0;
        for (int g = 0; g < n_grps_; ++g) {
            phi_adj_g.SetDataAndSize(phi_adj_n->GetData() + g * n_phi, n_phi);
            M_fiss_[g]->Mult(phi_adj_g, prod_g);
            production_new += prod_g.Sum();
        }
        
        double k_new = k_est * (production_new / production);
        diff_k = std::abs(k_new - k_est) / std::abs(k_est);
        k_est = k_new;
        
        // Normaliser
        norm = phi_adj_n->Norml2();
        if (norm > 1e-14) *phi_adj_n /= norm;
        
        if (diff_k < eigensolver_params_.tol_eigenvalue) {
            result.converged = true;
            break;
        }
    }
    
    double lambda_final = 1.0 / k_est;
    result.lambda = lambda_final;
    result.keff = k_est;
    result.iterations = iter;
    result.dominance_ratio = previous_eigenvalues.empty() ? 1.0 : 
                             lambda_final / previous_eigenvalues[0];
    
    eigenvectors_adj_.push_back(phi_adj_n);
    eigenvalues_adj_.push_back(lambda_final);
    
    return result;
}

// ============================================================================
// RÉSOLUTION DES MODES ADJOINTS
// ============================================================================

std::vector<EigenMode> NeutMFEM::SolveAdjointEigenmodes(
    int num_modes,
    LinearSolverType solver_type,
    const EigensolverParameters& params) {
    
    Log(VerbosityLevel::NORMAL, "\n=== CALCUL DES MODES PROPRES ADJOINTS ===");
    Log(VerbosityLevel::NORMAL, "Problème : L†·φ† = λ·F†·φ†");
    
    eigensolver_params_ = params;
    
    // Nettoyer les données adjointes
    for (auto* gf : eigenvectors_adj_) delete gf;
    eigenvectors_adj_.clear();
    eigenvalues_adj_.clear();
    
    std::vector<EigenMode> results;
    
    for (int n = 0; n < num_modes; ++n) {
        Log(VerbosityLevel::NORMAL, "\n--- Calcul du mode adjoint ", n, " ---");
        
        std::vector<mfem::GridFunction*> prev_modes;
        std::vector<double> prev_lambdas;
        for (int j = 0; j < n; ++j) {
            prev_modes.push_back(eigenvectors_adj_[j]);
            prev_lambdas.push_back(eigenvalues_adj_[j]);
        }
        
        EigenMode mode = ComputeSingleAdjointEigenmode(n, solver_type, prev_modes, prev_lambdas);
        results.push_back(mode);
        
        if (mode.converged) {
            Log(VerbosityLevel::NORMAL, "✓ Mode adjoint ", n, " : λ†_", n, " = ", mode.lambda);
        }
    }
    
    if (!eigenvectors_adj_.empty()) {
        *Sol_Phi_adj_gf_ = *eigenvectors_adj_[0];
    }
    
    return results;
}

// ============================================================================
// ACCESSEURS
// ============================================================================

double NeutMFEM::get_eigenvalue(int mode_index) const {
    if (mode_index < 0 || mode_index >= static_cast<int>(eigenvalues_.size())) {
        throw std::out_of_range("Index de mode invalide: " + std::to_string(mode_index));
    }
    return eigenvalues_[mode_index];
}

std::vector<double> NeutMFEM::get_keff_values() const {
    std::vector<double> keffs;
    keffs.reserve(eigenvalues_.size());
    for (double lambda : eigenvalues_) {
        keffs.push_back(1.0 / lambda);
    }
    return keffs;
}

py::array_t<double> NeutMFEM::get_eigenvector(int mode_index) {
    if (mode_index < 0 || mode_index >= static_cast<int>(eigenvectors_.size())) {
        throw std::out_of_range("Index de mode invalide: " + std::to_string(mode_index));
    }
    return GetCoefArray(eigenvectors_[mode_index]);
}

py::array_t<double> NeutMFEM::get_adjoint_eigenvector(int mode_index) {
    if (mode_index < 0 || mode_index >= static_cast<int>(eigenvectors_adj_.size())) {
        throw std::out_of_range("Index de mode adjoint invalide: " + std::to_string(mode_index));
    }
    return GetCoefArray(eigenvectors_adj_[mode_index]);
}

double NeutMFEM::GetDominanceRatio(int mode_index) const {
    if (mode_index < 0 || mode_index >= static_cast<int>(eigenvalues_.size())) {
        throw std::out_of_range("Index de mode invalide");
    }
    if (eigenvalues_.empty() || std::abs(eigenvalues_[0]) < 1e-14) {
        throw std::runtime_error("Eigenvalue fondamentale non disponible");
    }
    return eigenvalues_[mode_index] / eigenvalues_[0];
}

std::vector<std::vector<double>> NeutMFEM::CheckOrthogonality() const {
    const int n_modes = static_cast<int>(eigenvectors_.size());
    std::vector<std::vector<double>> matrix(n_modes, std::vector<double>(n_modes, 0.0));
    
    for (int i = 0; i < n_modes; ++i) {
        for (int j = 0; j < n_modes; ++j) {
            matrix[i][j] = const_cast<NeutMFEM*>(this)->FissionInnerProduct(
                *eigenvectors_[i], *eigenvectors_[j]);
        }
    }
    
    return matrix;
}

// ============================================================================
// EXPORT VTK
// ============================================================================

void NeutMFEM::SaveEigenvectorsVTK(const std::string& filename_prefix, int max_modes) {
    if (eigenvectors_.empty()) {
        Log(VerbosityLevel::LIGHT, "Aucun eigenvector à exporter");
        return;
    }
    
    const int n_modes = (max_modes <= 0 || max_modes > static_cast<int>(eigenvectors_.size())) 
                        ? static_cast<int>(eigenvectors_.size()) 
                        : max_modes;
    
    const int n_phi = fes_Phi_->GetVSize();
    
    Log(VerbosityLevel::NORMAL, "Export VTK de ", n_modes, " modes × ", n_grps_, " groupes...");
    
    // Créer le DataCollection
    mfem::ParaViewDataCollection pd(filename_prefix, mesh_);
    pd.SetPrefixPath("VTK_Eigenmodes");
    pd.SetLevelsOfDetail(1);
    pd.SetHighOrderOutput(false);
    pd.SetDataFormat(mfem::VTKFormat::BINARY);
    
    // Créer des GridFunctions mono-groupe temporaires pour chaque (mode, groupe)
    std::vector<mfem::GridFunction*> gf_exports;
    
    for (int m = 0; m < n_modes; ++m) {
        for (int g = 0; g < n_grps_; ++g) {
            // Créer une GridFunction mono-groupe
            mfem::GridFunction* gf_mg = new mfem::GridFunction(fes_Phi_);
            
            // Copier les données du groupe g depuis l'eigenvector m
            double* src = eigenvectors_[m]->GetData() + g * n_phi;
            double* dst = gf_mg->GetData();
            std::memcpy(dst, src, n_phi * sizeof(double));
            
            // Nom du champ : flux_harm_{mode}_{groupe}
            std::string field_name = "flux_harm_" + std::to_string(m) + "_" + std::to_string(g);
            
            pd.RegisterField(field_name, gf_mg);
            gf_exports.push_back(gf_mg);
        }
    }
    
    pd.SetCycle(0);
    pd.SetTime(0.0);
    pd.Save();
    
    // Nettoyage des GridFunctions temporaires
    for (auto* gf : gf_exports) {
        delete gf;
    }
    
    Log(VerbosityLevel::NORMAL, "Export terminé: ", filename_prefix);
}

} // namespace neutmfem


