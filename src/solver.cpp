/**
 * @file solver.cpp
 * @brief Implémentation des algorithmes d'accélération de convergence
 * 
 * @author jujuC31
 * @version 0.0.1
 * @date 2026
 */

#include "solver.hpp"
#include <cmath>
#include <stdexcept>
#include <algorithm>

namespace neutmfem{

// ============================================================================
// IMPLÉMENTATION CHEBYSHEV ACCELERATOR
// ============================================================================

ChebyshevAccelerator::ChebyshevAccelerator(int max_iterations, 
                                           double dominance_ratio)
    : max_iterations_(max_iterations)
    , iteration_(0)
    , dominance_ratio_(dominance_ratio) {

    // Validation des paramètres
    if (max_iterations <= 0) {
        throw std::invalid_argument(
            "ChebyshevAccelerator: max_iterations doit être positif");
    }
    if (dominance_ratio <= 0.0 || dominance_ratio >= 1.0) {
        throw std::invalid_argument(
            "ChebyshevAccelerator: dominance_ratio doit être dans ]0, 1[");
    }

    // Pré-calcul des coefficients de Chebyshev
    alpha_coeffs_.resize(max_iterations);
    beta_coeffs_.resize(max_iterations);

    // G = acosh(2/σ - 1) est le paramètre fondamental
    const double G = std::acosh(2.0 / dominance_ratio - 1.0);

    // Coefficient initial (itération 1)
    alpha_coeffs_[1] = 2.0 / (2.0 - dominance_ratio);

    // Coefficients pour les itérations supérieures
    for (int k = 2; k < max_iterations; ++k) {
        // α_k = cosh((k-1)G) / cosh(kG)
        alpha_coeffs_[k] = std::cosh((k - 1) * G) / std::cosh(k * G);
        // β_k = cosh((k-2)G) / cosh(kG)
        beta_coeffs_[k] = std::cosh((k - 2) * G) / std::cosh(k * G);
    }
}

void ChebyshevAccelerator::operator()(mfem::Vector& phi) {
    // Reset automatique si on dépasse le nombre max d'itérations
    if (iteration_ >= max_iterations_) {
        reset();
    }

    const int n = phi.Size();

    if (iteration_ == 0) {
        // Première itération: stockage simple sans accélération
        phi_prev2_ = phi;
        ++iteration_;
        return;
    }
    
    if (iteration_ == 1) {
        // Deuxième itération: interpolation linéaire simple
        phi_prev1_.SetSize(n);
        
        const double alpha = alpha_coeffs_[1];
        // φ_new = (1 - α)·φ_prev2 + α·φ
        phi_prev1_.Set(1.0 - alpha, phi_prev2_);
        phi_prev1_.Add(alpha, phi);
        
        phi = phi_prev1_;
        ++iteration_;
        return;
    }
    
    // Itérations ≥ 2: polynôme de Chebyshev complet
    mfem::Vector phi_new(n);
    
    const double alpha = alpha_coeffs_[iteration_];
    const double beta = beta_coeffs_[iteration_];
    const double factor = 4.0 / dominance_ratio_ * alpha;
    
    // φ_new = (1 - factor + β)·φ_prev1 + factor·φ - β·φ_prev2
    phi_new.Set(1.0 - factor + beta, phi_prev1_);
    phi_new.Add(factor, phi);
    phi_new.Add(-beta, phi_prev2_);
    
    // Mise à jour de l'historique avec move semantics
    phi_prev2_ = std::move(phi_prev1_);
    phi_prev1_ = std::move(phi_new);
    
    phi = phi_prev1_;
    ++iteration_;
}

void ChebyshevAccelerator::reset() {
    iteration_ = 0;
    // Libération de la mémoire
    phi_prev2_.SetSize(0);
    phi_prev1_.SetSize(0);
}

// ============================================================================
// IMPLÉMENTATION ANDERSON ACCELERATOR
// ============================================================================

AndersonAccelerator::AndersonAccelerator(int history_size, 
                                         double relaxation_factor)
    : max_history_(history_size)
    , relaxation_(relaxation_factor)
    , regularization_(1e-10)
    , max_relative_step_(0.3) {

    // Validation des paramètres
    if (history_size <= 0) {
        throw std::invalid_argument(
            "AndersonAccelerator: history_size doit être positif");
    }
    if (relaxation_factor <= 0.0 || relaxation_factor > 1.0) {
        throw std::invalid_argument(
            "AndersonAccelerator: relaxation_factor doit être dans ]0, 1]");
    }
}

bool AndersonAccelerator::solveCholesky(const std::vector<std::vector<double>>& A, 
                                         const std::vector<double>& b, 
                                         std::vector<double>& x,
                                         int n) {
    // Factorisation de Cholesky: A = L * L^T
    // Résolution: L * y = b, puis L^T * x = y
    
    std::vector<std::vector<double>> L(n, std::vector<double>(n, 0.0));
    
    // Factorisation de Cholesky
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j <= i; ++j) {
            double sum = A[i][j];
            
            for (int k = 0; k < j; ++k) {
                sum -= L[i][k] * L[j][k];
            }
            
            if (i == j) {
                // Élément diagonal
                if (sum <= 0.0) {
                    // Matrice non définie positive
                    return false;
                }
                L[i][j] = std::sqrt(sum);
            } else {
                // Élément hors diagonale
                if (std::abs(L[j][j]) < 1e-14) {
                    return false;
                }
                L[i][j] = sum / L[j][j];
            }
        }
    }
    
    // Résolution de L * y = b (substitution avant)
    std::vector<double> y(n);
    for (int i = 0; i < n; ++i) {
        double sum = b[i];
        for (int j = 0; j < i; ++j) {
            sum -= L[i][j] * y[j];
        }
        if (std::abs(L[i][i]) < 1e-14) {
            return false;
        }
        y[i] = sum / L[i][i];
    }
    
    // Résolution de L^T * x = y (substitution arrière)
    x.resize(n);
    for (int i = n - 1; i >= 0; --i) {
        double sum = y[i];
        for (int j = i + 1; j < n; ++j) {
            sum -= L[j][i] * x[j];  // L^T[i][j] = L[j][i]
        }
        if (std::abs(L[i][i]) < 1e-14) {
            return false;
        }
        x[i] = sum / L[i][i];
    }
    
    return true;
}

void AndersonAccelerator::apply(mfem::Vector& phi, mfem::Vector& out) {
    const int N = phi.Size();

    // Première itération: pas d'accélération possible
    if (x_history_.empty()) {
        x_history_.push_back(phi);
        f_history_.emplace_back(N);
        f_history_.back() = 0.0;
        out = phi;
        return;
    }

    // Calcul du résidu: f_new = phi - x_history.back()
    mfem::Vector f_new(N);
    add(phi, -1.0, x_history_.back(), f_new);

    // Mise à jour de l'historique
    x_history_.push_back(phi);
    f_history_.push_back(f_new);

    // Limitation de la taille de l'historique
    while (x_history_.size() > static_cast<size_t>(max_history_)) {
        x_history_.pop_front();
        f_history_.pop_front();
    }

    const int m = static_cast<int>(f_history_.size());

    // Historique trop petit: pas d'accélération
    if (m <= 1) {
        out = phi;
        return;
    }

    // Construction du problème des moindres carrés
    // min ||F·α - f_new||² où F = [Δf_0, Δf_1, ..., Δf_{m-2}]
    const int num_coeffs = m - 1;
    
    // Pré-calcul des différences Δf et Δx
    std::vector<mfem::Vector> delta_f(num_coeffs);
    std::vector<mfem::Vector> delta_x(num_coeffs);

    for (int i = 0; i < num_coeffs; ++i) {
        delta_f[i].SetSize(N);
        add(f_history_[i + 1], -1.0, f_history_[i], delta_f[i]);

        delta_x[i].SetSize(N);
        add(x_history_[i + 1], -1.0, x_history_[i], delta_x[i]);
    }

    // Vecteur RHS pour le système normal: rhs_diff = f_new - f_{m-2}
    mfem::Vector rhs_diff(N);
    add(f_new, -1.0, f_history_[m - 2], rhs_diff);

    // Construction de F^T·F et F^T·rhs (avec std::vector)
    std::vector<std::vector<double>> FtF(num_coeffs, std::vector<double>(num_coeffs, 0.0));
    std::vector<double> rhs_vec(num_coeffs, 0.0);

    for (int i = 0; i < num_coeffs; ++i) {
        for (int j = 0; j < num_coeffs; ++j) {
            FtF[i][j] = mfem::InnerProduct(delta_f[i], delta_f[j]);
        }
        rhs_vec[i] = mfem::InnerProduct(delta_f[i], rhs_diff);
    }

    // Régularisation de Tikhonov pour stabilité numérique
    for (int i = 0; i < num_coeffs; ++i) {
        FtF[i][i] += regularization_;
    }

    // Résolution par décomposition de Cholesky
    std::vector<double> alpha(num_coeffs);
    bool success = solveCholesky(FtF, rhs_vec, alpha, num_coeffs);
    
    if (!success) {
        // Échec de Cholesky: pas d'accélération
        out = phi;
        return;
    }

    // Construction de la correction: Δx = Σ α_i · Δx_i
    mfem::Vector correction(N);
    correction = 0.0;

    for (int i = 0; i < num_coeffs; ++i) {
        correction.Add(alpha[i], delta_x[i]);
    }

    // Clamping de la correction pour éviter les oscillations
    const double phi_norm = phi.Norml2();
    const double correction_norm = correction.Norml2();

    if (phi_norm > 1e-12 && (correction_norm / phi_norm) > max_relative_step_) {
        correction *= (max_relative_step_ * phi_norm / correction_norm);
    }

    // Application de la correction avec relaxation
    // out = φ - β·correction
    out.Set(1.0, phi);
    out.Add(-relaxation_, correction);
}

void AndersonAccelerator::reset() {
    x_history_.clear();
    f_history_.clear();
}

// ============================================================================
// IMPLÉMENTATION SOLVEUR DE THOMAS
// ============================================================================

void thomas_solve(const mfem::Vector& lower_diag,
                  const mfem::Vector& main_diag,
                  const mfem::Vector& upper_diag,
                  const mfem::Vector& rhs,
                  mfem::Vector& solution) {

    const int n = rhs.Size();

    // Validation des dimensions
    if (lower_diag.Size() != n || main_diag.Size() != n || upper_diag.Size() != n) {
        throw std::invalid_argument(
            "thomas_solve: dimensions incompatibles des vecteurs");
    }

    solution.SetSize(n);

    // Allocation des vecteurs de travail pour la factorisation LU
    std::vector<double> c_prime(n);  // Diagonale supérieure modifiée
    std::vector<double> d_prime(n);  // Second membre modifié

    // ========================================
    // Phase 1: Élimination avant (Forward sweep)
    // ========================================
    
    if (std::abs(main_diag(0)) < 1e-14) {
        throw std::runtime_error(
            "thomas_solve: matrice singulière (pivot nul en position 0)");
    }
    
    c_prime[0] = upper_diag(0) / main_diag(0);
    d_prime[0] = rhs(0) / main_diag(0);

    for (int i = 1; i < n; ++i) {
        const double denom = main_diag(i) - lower_diag(i) * c_prime[i - 1];

        if (std::abs(denom) < 1e-14) {
            throw std::runtime_error(
                "thomas_solve: matrice singulière ou mal conditionnée "
                "(pivot nul en position " + std::to_string(i) + ")");
        }

        if (i < n - 1) {
            c_prime[i] = upper_diag(i) / denom;
        }

        d_prime[i] = (rhs(i) - lower_diag(i) * d_prime[i - 1]) / denom;
    }

    // ========================================
    // Phase 2: Substitution arrière (Backward substitution)
    // ========================================
    
    solution(n - 1) = d_prime[n - 1];

    for (int i = n - 2; i >= 0; --i) {
        solution(i) = d_prime[i] - c_prime[i] * solution(i + 1);
    }
}

}
