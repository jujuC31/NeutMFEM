/**
 * @file solver.hpp
 * @brief Algorithmes d'accélération pour les itérations de puissance neutroniques
 * 
 * Ce fichier contient les implémentations des accélérateurs de convergence
 * utilisés dans la résolution des équations de diffusion neutronique :
 * - Accélération de Chebyshev (polynomiale)
 * - Accélération d'Anderson (moindres carrés)
 * - Solveur de Thomas (systèmes tridiagonaux)
 * 
 * @author jujuc31
 * @version 0.0.1
 * @date 2026
 * 
 * @note Dépendances: MFEM et pybind11
 */

#ifndef SOLVER_HPP
#define SOLVER_HPP

#include "mfem.hpp"
#include <deque>
#include <vector>

namespace neutmfem{

// ============================================================================
// ACCÉLÉRATION DE CHEBYSHEV
// ============================================================================

/**
 * @class ChebyshevAccelerator
 * @brief Accélérateur de Chebyshev pour les itérations de puissance
 * 
 * Améliore la convergence des itérations de puissance en utilisant
 * des polynômes de Chebyshev pour accélérer la séquence d'itérations.
 */
class ChebyshevAccelerator {
public:
    /**
     * @brief Constructeur avec paramètres d'accélération
     * 
     * @param max_iterations Nombre maximum d'itérations avant reset automatique
     * @param dominance_ratio Estimation du ratio de dominance spectrale σ = λ₂/λ₁
     * 
     * @throws std::invalid_argument Si max_iterations ≤ 0 ou dominance_ratio ∉ ]0,1[
     */
    explicit ChebyshevAccelerator(int max_iterations = 10, 
                                   double dominance_ratio = 0.90);

    /**
     * @brief Applique l'accélération de Chebyshev au vecteur flux
     * @param[in,out] phi Vecteur flux à accélérer (modifié en place)
     */
    void operator()(mfem::Vector& phi);

    /**
     * @brief Réinitialise l'accélérateur à son état initial
     */
    void reset();

    /**
     * @brief Retourne le numéro de l'itération courante
     */
    [[nodiscard]] int current_iteration() const noexcept { return iteration_; }

private:
    int max_iterations_;
    int iteration_;
    double dominance_ratio_;

    std::vector<double> alpha_coeffs_;
    std::vector<double> beta_coeffs_;

    mfem::Vector phi_prev2_;
    mfem::Vector phi_prev1_;
};

// ============================================================================
// ACCÉLÉRATION D'ANDERSON
// ============================================================================

/**
 * @class AndersonAccelerator
 * @brief Accélérateur d'Anderson (AA) pour les itérations à point fixe
 * 
 * Méthode d'accélération basée sur la minimisation aux moindres carrés
 * du résidu. Cette implémentation n'utilise PAS Eigen et résout le
 * système normal directement avec une factorisation Cholesky via MFEM.
 * 
 * @par Algorithme
 * À chaque itération, on cherche les coefficients α qui minimisent:
 * @f[
 *   \min_{\alpha} \| \sum_{i=0}^{m-1} \alpha_i f_{k-i} \|^2 
 * @f]
 * 
 * Le système normal F^T F α = F^T f_new est résolu par Cholesky.
 */
class AndersonAccelerator {
public:
    /**
     * @brief Constructeur avec paramètres d'accélération
     * 
     * @param history_size Taille de l'historique m (défaut: 5, recommandé: 3-10)
     * @param relaxation_factor Facteur de relaxation β ∈ ]0, 1] (défaut: 0.8)
     * 
     * @throws std::invalid_argument Si history_size ≤ 0 ou relaxation_factor ∉ ]0,1]
     */
    explicit AndersonAccelerator(int history_size = 5, 
                                  double relaxation_factor = 0.8);

    /**
     * @brief Applique l'accélération d'Anderson
     * 
     * @param[in] phi_current Vecteur courant de l'itération
     * @param[out] phi_accelerated Vecteur accéléré en sortie
     */
    void apply(mfem::Vector& phi_current, mfem::Vector& phi_accelerated);

    /**
     * @brief Réinitialise l'historique de l'accélérateur
     */
    void reset();

    /**
     * @brief Retourne la taille actuelle de l'historique
     */
    [[nodiscard]] size_t current_history_size() const noexcept { 
        return x_history_.size(); 
    }

private:
    int max_history_;
    double relaxation_;
    double regularization_;
    double max_relative_step_;

    std::deque<mfem::Vector> x_history_;
    std::deque<mfem::Vector> f_history_;
    
    /**
     * @brief Résout le système symétrique défini positif par Cholesky
     * 
     * Implémentation manuelle sans dépendance externe.
     * 
     * @param A Matrice symétrique définie positive (n x n)
     * @param b Second membre (n)
     * @param x Solution (n)
     * @param n Dimension du système
     * @return true si succès, false si matrice singulière
     */
    bool solveCholesky(const std::vector<std::vector<double>>& A, 
                       const std::vector<double>& b, 
                       std::vector<double>& x,
                       int n);
};

// ============================================================================
// SOLVEUR DE THOMAS (SYSTÈMES TRIDIAGONAUX)
// ============================================================================

/**
 * @brief Résout un système tridiagonal A·x = rhs par l'algorithme de Thomas
 * 
 * Algorithme de factorisation LU optimisé pour les matrices tridiagonales.
 * Complexité O(n) en temps et en mémoire auxiliaire.
 * 
 * @param[in] lower_diag Diagonale inférieure a (taille n, lower_diag[0] ignoré)
 * @param[in] main_diag Diagonale principale b (taille n)
 * @param[in] upper_diag Diagonale supérieure c (taille n, upper_diag[n-1] ignoré)
 * @param[in] rhs Second membre (taille n)
 * @param[out] solution Solution x (taille n)
 * 
 * @throws std::invalid_argument Si les tailles des vecteurs sont incompatibles
 * @throws std::runtime_error Si la matrice est singulière
 */
void thomas_solve(const mfem::Vector& lower_diag,
                  const mfem::Vector& main_diag,
                  const mfem::Vector& upper_diag,
                  const mfem::Vector& rhs,
                  mfem::Vector& solution);

// ============================================================================
// ALIAS DE COMPATIBILITÉ (DÉPRÉCIÉS)
// ============================================================================

/// @deprecated Utilisez ChebyshevAccelerator à la place
using TchebyAccel = ChebyshevAccelerator;

/// @deprecated Utilisez AndersonAccelerator à la place  
using AndersonAccel = AndersonAccelerator;

/// @deprecated Utilisez thomas_solve à la place
inline void ThomasSolver(const mfem::Vector& a, const mfem::Vector& b,
                         const mfem::Vector& c, const mfem::Vector& rhs,
                         mfem::Vector& x) {
    thomas_solve(a, b, c, rhs, x);
}

} // namespace neutmfem

#endif // SOLVER_HPP
