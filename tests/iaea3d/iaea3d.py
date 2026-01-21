"""
BENCHMARK IAEA 3D - Version adaptée pour NeutFEM
Benchmark 3D - 2 groupes d'énergie avec réflecteur
"""

import time
import numpy as np
import argparse

import neutmfem._neutmfem as neutron_solver
from neutmfem._neutmfem import BCType, BoundaryAttribute, LinearSolverType

import seaborn as sns
import matplotlib.pyplot as plt

np.set_printoptions(threshold=np.inf, linewidth=np.inf)


class Iaea3D:
    """
    Benchmark IAEA 3D avec fonctionnalités avancées
    """
    
    def __init__(self, meshtype="2x2", domaine="entier", ncpu=1, sym="cyclique"):
        
        self.start = time.time()
        self.meshtype = meshtype
        self.domaine = domaine
        self.ncpu = ncpu
        self.sym = sym
        self.nmeshes_z = 1
        
        self.kref = 1.029096  # k-eff référence
        self.num_groups = 2
        self.verbose = 0
        self.order = 0
        
        self.init_meshing = False
        self.mysolv = None
        self.keff = None
        self.phi = None
        self.pvol = None

        # Configuration axiale (19 plans axiaux)
        # FA = Plan avec réflecteur uniquement
        FA = np.array([
            ["  ", "  ", "  ", "  ", "  ", "  ", "  ", "  ", "  ", "  ", "  ", "  ", "  ", "  ", "  ", "  ", "  ", "  ", "  "],
            ["  ", "  ", "  ", "  ", "  ", "  ", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "  ", "  ", "  ", "  ", "  ", "  "],
            ["  ", "  ", "  ", "  ", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "  ", "  ", "  ", "  "],
            ["  ", "  ", "  ", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "  ", "  ", "  "],
            ["  ", "  ", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "  ", "  "],
            ["  ", "  ", "F4", "F4", "F4", "F5", "F4", "F4", "F4", "F5", "F4", "F4", "F4", "F5", "F4", "F4", "F4", "  ", "  "],
            ["  ", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "  "],
            ["  ", "F4", "F4", "F4", "F4", "F4", "F4", "F5", "F4", "F4", "F4", "F5", "F4", "F4", "F4", "F4", "F4", "F4", "  "],
            ["  ", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "  "],
            ["  ", "F4", "F4", "F4", "F4", "F5", "F4", "F4", "F4", "F5", "F4", "F4", "F4", "F5", "F4", "F4", "F4", "F4", "  "],
            ["  ", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "  "],
            ["  ", "F4", "F4", "F4", "F4", "F4", "F4", "F5", "F4", "F4", "F4", "F5", "F4", "F4", "F4", "F4", "F4", "F4", "  "],
            ["  ", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "  "],
            ["  ", "  ", "F4", "F4", "F4", "F5", "F4", "F4", "F4", "F5", "F4", "F4", "F4", "F5", "F4", "F4", "F4", "  ", "  "],
            ["  ", "  ", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "  ", "  "],
            ["  ", "  ", "  ", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "  ", "  ", "  "],
            ["  ", "  ", "  ", "  ", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "  ", "  ", "  ", "  "],
            ["  ", "  ", "  ", "  ", "  ", "  ", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "  ", "  ", "  ", "  ", "  ", "  "],
            ["  ", "  ", "  ", "  ", "  ", "  ", "  ", "  ", "  ", "  ", "  ", "  ", "  ", "  ", "  ", "  ", "  ", "  ", "  "]
        ])

        # FB = Plan avec combustible et barres de contrôle
        FB = np.array([
            ["  ", "  ", "  ", "  ", "  ", "  ", "  ", "  ", "  ", "  ", "  ", "  ", "  ", "  ", "  ", "  ", "  ", "  ", "  "],
            ["  ", "  ", "  ", "  ", "  ", "  ", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "  ", "  ", "  ", "  ", "  ", "  "],
            ["  ", "  ", "  ", "  ", "F4", "F4", "F4", "F3", "F3", "F3", "F3", "F3", "F4", "F4", "F4", "  ", "  ", "  ", "  "],
            ["  ", "  ", "  ", "F4", "F4", "F3", "F3", "F3", "F1", "F1", "F1", "F3", "F3", "F3", "F4", "F4", "  ", "  ", "  "],
            ["  ", "  ", "F4", "F4", "F3", "F3", "F1", "F1", "F1", "F1", "F1", "F1", "F1", "F3", "F3", "F4", "F4", "  ", "  "],
            ["  ", "  ", "F4", "F3", "F3", "F2", "F1", "F1", "F1", "F2", "F1", "F1", "F1", "F2", "F3", "F3", "F4", "  ", "  "],
            ["  ", "F4", "F4", "F3", "F1", "F1", "F1", "F1", "F1", "F1", "F1", "F1", "F1", "F1", "F1", "F3", "F4", "F4", "  "],
            ["  ", "F4", "F3", "F3", "F1", "F1", "F1", "F2", "F1", "F1", "F1", "F2", "F1", "F1", "F1", "F3", "F3", "F4", "  "],
            ["  ", "F4", "F3", "F1", "F1", "F1", "F1", "F1", "F1", "F1", "F1", "F1", "F1", "F1", "F1", "F1", "F3", "F4", "  "],
            ["  ", "F4", "F3", "F1", "F1", "F2", "F1", "F1", "F1", "F2", "F1", "F1", "F1", "F2", "F1", "F1", "F3", "F4", "  "],
            ["  ", "F4", "F3", "F1", "F1", "F1", "F1", "F1", "F1", "F1", "F1", "F1", "F1", "F1", "F1", "F1", "F3", "F4", "  "],
            ["  ", "F4", "F3", "F3", "F1", "F1", "F1", "F2", "F1", "F1", "F1", "F2", "F1", "F1", "F1", "F3", "F3", "F4", "  "],
            ["  ", "F4", "F4", "F3", "F1", "F1", "F1", "F1", "F1", "F1", "F1", "F1", "F1", "F1", "F1", "F3", "F4", "F4", "  "],
            ["  ", "  ", "F4", "F3", "F3", "F2", "F1", "F1", "F1", "F2", "F1", "F1", "F1", "F2", "F3", "F3", "F4", "  ", "  "],
            ["  ", "  ", "F4", "F4", "F3", "F3", "F1", "F1", "F1", "F1", "F1", "F1", "F1", "F3", "F3", "F4", "F4", "  ", "  "],
            ["  ", "  ", "  ", "F4", "F4", "F3", "F3", "F3", "F1", "F1", "F1", "F3", "F3", "F3", "F4", "F4", "  ", "  ", "  "],
            ["  ", "  ", "  ", "  ", "F4", "F4", "F4", "F3", "F3", "F3", "F3", "F3", "F4", "F4", "F4", "  ", "  ", "  ", "  "],
            ["  ", "  ", "  ", "  ", "  ", "  ", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "  ", "  ", "  ", "  ", "  ", "  "],
            ["  ", "  ", "  ", "  ", "  ", "  ", "  ", "  ", "  ", "  ", "  ", "  ", "  ", "  ", "  ", "  ", "  ", "  ", "  "]
        ])

        # FC = Plan avec combustible sans barres de contrôle
        FC = np.array([
            ["  ", "  ", "  ", "  ", "  ", "  ", "  ", "  ", "  ", "  ", "  ", "  ", "  ", "  ", "  ", "  ", "  ", "  ", "  "],
            ["  ", "  ", "  ", "  ", "  ", "  ", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "  ", "  ", "  ", "  ", "  ", "  "],
            ["  ", "  ", "  ", "  ", "F4", "F4", "F4", "F3", "F3", "F3", "F3", "F3", "F4", "F4", "F4", "  ", "  ", "  ", "  "],
            ["  ", "  ", "  ", "F4", "F4", "F3", "F3", "F3", "F1", "F1", "F1", "F3", "F3", "F3", "F4", "F4", "  ", "  ", "  "],
            ["  ", "  ", "F4", "F4", "F3", "F3", "F1", "F1", "F1", "F1", "F1", "F1", "F1", "F3", "F3", "F4", "F4", "  ", "  "],
            ["  ", "  ", "F4", "F3", "F3", "F2", "F1", "F1", "F1", "F2", "F1", "F1", "F1", "F2", "F3", "F3", "F4", "  ", "  "],
            ["  ", "F4", "F4", "F3", "F1", "F1", "F1", "F1", "F1", "F1", "F1", "F1", "F1", "F1", "F1", "F3", "F4", "F4", "  "],
            ["  ", "F4", "F3", "F3", "F1", "F1", "F1", "F1", "F1", "F1", "F1", "F1", "F1", "F1", "F1", "F3", "F3", "F4", "  "],
            ["  ", "F4", "F3", "F1", "F1", "F1", "F1", "F1", "F1", "F1", "F1", "F1", "F1", "F1", "F1", "F1", "F3", "F4", "  "],
            ["  ", "F4", "F3", "F1", "F1", "F2", "F1", "F1", "F1", "F2", "F1", "F1", "F1", "F2", "F1", "F1", "F3", "F4", "  "],
            ["  ", "F4", "F3", "F1", "F1", "F1", "F1", "F1", "F1", "F1", "F1", "F1", "F1", "F1", "F1", "F1", "F3", "F4", "  "],
            ["  ", "F4", "F3", "F3", "F1", "F1", "F1", "F1", "F1", "F1", "F1", "F1", "F1", "F1", "F1", "F3", "F3", "F4", "  "],
            ["  ", "F4", "F4", "F3", "F1", "F1", "F1", "F1", "F1", "F1", "F1", "F1", "F1", "F1", "F1", "F3", "F4", "F4", "  "],
            ["  ", "  ", "F4", "F3", "F3", "F2", "F1", "F1", "F1", "F2", "F1", "F1", "F1", "F2", "F3", "F3", "F4", "  ", "  "],
            ["  ", "  ", "F4", "F4", "F3", "F3", "F1", "F1", "F1", "F1", "F1", "F1", "F1", "F3", "F3", "F4", "F4", "  ", "  "],
            ["  ", "  ", "  ", "F4", "F4", "F3", "F3", "F3", "F1", "F1", "F1", "F3", "F3", "F3", "F4", "F4", "  ", "  ", "  "],
            ["  ", "  ", "  ", "  ", "F4", "F4", "F4", "F3", "F3", "F3", "F3", "F3", "F4", "F4", "F4", "  ", "  ", "  ", "  "],
            ["  ", "  ", "  ", "  ", "  ", "  ", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "  ", "  ", "  ", "  ", "  ", "  "],
            ["  ", "  ", "  ", "  ", "  ", "  ", "  ", "  ", "  ", "  ", "  ", "  ", "  ", "  ", "  ", "  ", "  ", "  ", "  "]
        ])

        # FD = Plan réflecteur inférieur
        FD = np.array([
            ["  ", "  ", "  ", "  ", "  ", "  ", "  ", "  ", "  ", "  ", "  ", "  ", "  ", "  ", "  ", "  ", "  ", "  ", "  "],
            ["  ", "  ", "  ", "  ", "  ", "  ", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "  ", "  ", "  ", "  ", "  ", "  "],
            ["  ", "  ", "  ", "  ", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "  ", "  ", "  ", "  "],
            ["  ", "  ", "  ", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "  ", "  ", "  "],
            ["  ", "  ", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "  ", "  "],
            ["  ", "  ", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "  ", "  "],
            ["  ", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "  "],
            ["  ", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "  "],
            ["  ", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "  "],
            ["  ", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "  "],
            ["  ", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "  "],
            ["  ", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "  "],
            ["  ", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "  "],
            ["  ", "  ", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "  ", "  "],
            ["  ", "  ", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "  ", "  "],
            ["  ", "  ", "  ", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "  ", "  ", "  "],
            ["  ", "  ", "  ", "  ", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "  ", "  ", "  ", "  "],
            ["  ", "  ", "  ", "  ", "  ", "  ", "F4", "F4", "F4", "F4", "F4", "F4", "F4", "  ", "  ", "  ", "  ", "  ", "  "],
            ["  ", "  ", "  ", "  ", "  ", "  ", "  ", "  ", "  ", "  ", "  ", "  ", "  ", "  ", "  ", "  ", "  ", "  ", "  "]
        ])

        # Structure axiale: FA(1) + FB(4) + FC(13) + FD(1) = 19 plans
        self.maillage_motifs_coeur = [FA, FB, FB, FB, FB, FC, FC, FC, FC, FC, FC, FC, FC, FC, FC, FC, FC, FC, FD]

    def plot_geom(self, z_level=0):
        """Visualise la géométrie à un niveau z donné"""
        if not self.init_meshing:
            print("❌ Erreur: Le maillage doit être initialisé d'abord")
            return

        maillage_draw = []
        for row in self.maillage[z_level]:
            maillage_draw.append([
                0 if cell == "  " else int(cell[1]) 
                for cell in row
            ])

        sns.heatmap(maillage_draw, cmap='jet', linewidths=0.5, linecolor="k")
        plt.title(f"Géométrie - {self.meshtype} - {self.domaine} - Plan Z={z_level}")
        plt.show()

    def plot_materials(self):
        """Visualise les matériaux du cœur (plan central)"""
        plan_central = len(self.maillage_motifs_coeur) // 2
        maillage_draw = []
        for row in self.maillage_motifs_coeur[plan_central]:
            maillage_draw.append([
                0 if cell == "  " else int(cell[1]) 
                for cell in row
            ])

        sns.heatmap(maillage_draw, cmap='jet', annot=True, 
                   linewidths=1, linecolor="k", fmt='d')
        plt.title(f"Distribution des matériaux IAEA 3D (plan {plan_central})")
        plt.show()

    def mesh_initialisation(self, meshtype=None, domaine=None):
        """Initialise le maillage 3D"""
        timeref = time.time()

        if meshtype:
            self.meshtype = meshtype
        if domaine:
            self.domaine = domaine
        
        if "x" in self.meshtype:
            self.nmeshes = int(self.meshtype.split("x")[0])
        else:
            self.nmeshes = len(self.meshtype)

        # Expansion 3D
        self.maillage = np.array([
            [[cell for cell in row for _ in range(self.nmeshes)] 
             for row in zcell 
             for _ in range(self.nmeshes)] 
            for zcell in self.maillage_motifs_coeur 
            for _ in range(self.nmeshes_z)
        ])

        # Symétries (appliquées dans le plan XY)
        Nz = len(self.maillage)
        Ny = len(self.maillage[0])
        Nx = len(self.maillage[0][0])
        
        Ny_half = Ny // 2
        Nx_half = Nx // 2
        
        if self.domaine == "quart_so":
            tmp_maillage = np.empty((Nz, Ny_half, Nx_half), dtype='<U2')
            for k in range(Nz):
                tmp_maillage[k] = self.maillage[k][Ny_half:, :Nx_half]
            self.maillage = tmp_maillage
        
        elif self.domaine == "quart_no":
            tmp_maillage = np.empty((Nz, Ny_half, Nx_half), dtype='<U2')
            for k in range(Nz):
                tmp_maillage[k] = self.maillage[k][:Ny_half, :Nx_half]
            self.maillage = tmp_maillage
        
        elif self.domaine == "quart_ne":
            tmp_maillage = np.empty((Nz, Ny_half, Nx_half), dtype='<U2')
            for k in range(Nz):
                tmp_maillage[k] = self.maillage[k][:Ny_half, Nx_half:]
            self.maillage = tmp_maillage
        
        elif self.domaine == "quart_se":
            tmp_maillage = np.empty((Nz, Ny_half, Nx_half), dtype='<U2')
            for k in range(Nz):
                tmp_maillage[k] = self.maillage[k][Ny_half:, Nx_half:]
            self.maillage = tmp_maillage

        # Coordonnées
        cell_size = 20.0 / self.nmeshes
        cell_size_z = 20.0 / self.nmeshes_z
        
        nz_cells = self.maillage.shape[0]
        ny_cells = self.maillage.shape[1]
        nx_cells = self.maillage.shape[2]

        self.x_breaks = np.linspace(0.0, nx_cells * cell_size, nx_cells + 1)
        self.y_breaks = np.linspace(0.0, ny_cells * cell_size, ny_cells + 1)
        self.z_breaks = np.linspace(0.0, nz_cells * cell_size_z, nz_cells + 1)
        
        time1 = time.time()
        print(f"✅ Maillage initialisé: {ny_cells}×{nx_cells}×{nz_cells} cellules "
              f"({time1-timeref:.3f} s)")
        self.init_meshing = True

    def load_iaea3d_mat(self, include_upscattering=False):
        """Charge les matériaux avec upscattering optionnel"""
        
        # Matériau F1 - Combustible type 1
        self.F1 = {
            'D': [1.5, 0.4],
            'ABS': [0.010, 0.085],
            'NSF': [0.0, 0.135],
            'CHI': [1., 0.],
            'S12': 0.02
        }
        self.F1['SIGR'] = [self.F1['ABS'][0] + self.F1['S12'], self.F1['ABS'][1]]
        self.F1['S21'] = 0.0

        # Matériau F2 - Combustible type 2 avec barre de contrôle
        self.F2 = {
            'D': [1.5, 0.4],
            'ABS': [0.010, 0.130],
            'NSF': [0.0, 0.135],
            'CHI': [1., 0.],
            'S12': 0.02
        }
        self.F2['SIGR'] = [self.F2['ABS'][0] + self.F2['S12'], self.F2['ABS'][1]]
        self.F2['S21'] = 0.0

        # Matériau F3 - Combustible type 3
        self.F3 = {
            'D': [1.5, 0.4],
            'ABS': [0.010, 0.080],
            'NSF': [0.0, 0.135],
            'CHI': [1., 0.],
            'S12': 0.02
        }
        self.F3['SIGR'] = [self.F3['ABS'][0] + self.F3['S12'], self.F3['ABS'][1]]
        self.F3['S21'] = 0.0

        # Matériau F4 - Réflecteur
        self.F4 = {
            'D': [2.0, 0.3],
            'ABS': [0.000, 0.0100],
            'NSF': [0.0, 0.0],
            'CHI': [0.0, 0.0],
            'S12': 0.04
        }
        self.F4['SIGR'] = [self.F4['ABS'][0] + self.F4['S12'], self.F4['ABS'][1]]
        self.F4['S21'] = 0.0

        # Matériau F5 - Combustible sans barre (identique à F1)
        self.F5 = {
            'D': [2.0, 0.3],
            'ABS': [0.000, 0.0550],
            'NSF': [0.0, 0.0],
            'CHI': [0., 0.],
            'S12': 0.04
        }
        self.F5['SIGR'] = [self.F5['ABS'][0] + self.F5['S12'], self.F5['ABS'][1]]
        self.F5['S21'] = 0.0

        # Matériau R0 - Réflecteur externe
        self.R0 = self.F4.copy()

        if include_upscattering:
            print("⚠️  Upscattering activé (non implémenté pour IAEA 3D)")

    def init_solver(self):
        """Initialise le solveur neutronique 3D"""
        if not self.init_meshing:
            raise RuntimeError("Le maillage doit être initialisé avant le solveur")
        
        timeref = time.time()
        
        Nz, Ny, Nx = self.maillage.shape
        print(f"\n=== INITIALISATION SOLVEUR 3D ({Ny}×{Nx}×{Nz}, {self.num_groups} groupes) ===")
        
        # Création du solveur 3D
        self.mysolv = neutron_solver.NeutMFEM(
            self.order,
            self.num_groups,
            self.x_breaks,
            self.y_breaks,
            self.z_breaks
        )
        
        # Conditions aux limites par défaut
        self.mysolv.set_bc(BoundaryAttribute.LEFT_3D, BCType.DIRICHLET, 0.0)
        self.mysolv.set_bc(BoundaryAttribute.RIGHT_3D, BCType.DIRICHLET, 0.0)
        self.mysolv.set_bc(BoundaryAttribute.FRONT_3D, BCType.DIRICHLET, 0.0)
        self.mysolv.set_bc(BoundaryAttribute.BACK_3D, BCType.DIRICHLET, 0.0)
        self.mysolv.set_bc(BoundaryAttribute.BOTTOM_3D, BCType.DIRICHLET, 0.0)
        self.mysolv.set_bc(BoundaryAttribute.TOP_3D, BCType.DIRICHLET, 0.0)

        # Application des symétries dans le plan XY
        if self.domaine == "quart_so":
            self.mysolv.apply_central_symmetry(1, 1)
            self.mysolv.set_bc(BoundaryAttribute.FRONT_3D, BCType.MIRROR)
            self.mysolv.set_bc(BoundaryAttribute.RIGHT_3D, BCType.MIRROR)
            self.mysolv.set_bc(BoundaryAttribute.LEFT_3D, BCType.DIRICHLET, 0.0)
            self.mysolv.set_bc(BoundaryAttribute.BACK_3D, BCType.DIRICHLET, 0.0)
            print("  Domaine QUART_SO : Symétrie centrale activée")

        # Remplissage des coefficients
        print(f"\n=== REMPLISSAGE DES COEFFICIENTS ({Nz}×{Ny}×{Nx}) ===")
        
        for k in range(Nz):
            for i in range(Ny):
                for j in range(Nx):
                    fuel_key = self.maillage[k, i, j].strip()
                    mat = getattr(self, fuel_key) if hasattr(self, fuel_key) and fuel_key else self.R0
                    
                    for g in range(self.num_groups):
                        self.mysolv.get_D()[g, k, i, j] = mat['D'][g]
                        self.mysolv.get_NSF()[g, k, i, j] = mat['NSF'][g]
                        self.mysolv.get_Chi()[g, k, i, j] = mat['CHI'][g]
                        self.mysolv.get_SigR()[g, k, i, j] = mat['SIGR'][g]
                    
                    self.mysolv.get_SigS()[1, 0, k, i, j] = mat['S12']
                    self.mysolv.get_SigS()[0, 1, k, i, j] = mat['S21']
        
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
            Nz, Ny, Nx = self.maillage.shape
            self.pvol = np.zeros((Nz, Ny, Nx))
            
            for k in range(Nz):
                for i in range(Ny):
                    for j in range(Nx):
                        for g in range(self.num_groups):
                            nsf_g = self.mysolv.get_NSF()[g]
                            phi_g = self.mysolv.get_flux()[g]
                            self.pvol[k, i, j] += nsf_g[k, i, j] * phi_g[k, i, j]

    def plot_flux(self, group=0, z_level=None):
        """Visualise le flux à un niveau z donné"""
        if self.phi is None:
            print("❌ Flux non disponible")
            return

        if fine : 
            flux = self.phi_fins
        else : 
            flux = self.phi
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(self.phi[group, z_level], cmap='jet', 
                   cbar_kws={'label': f'Flux φ{group+1}'})
        plt.title(f"Flux Groupe {group+1} - Plan Z={z_level} - k-eff = {self.keff:.5f}")
        plt.tight_layout()
        plt.show()

    def plot_pvol(self, z_level=None):
        """Visualise la puissance à un niveau z donné"""
        if self.pvol is None:
            print("❌ Puissance non disponible")
            return
        
        if z_level is None:
            z_level = self.pvol.shape[0] // 2  # Plan central
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(self.pvol[z_level], cmap='jet', 
                   cbar_kws={'label': 'Puissance'})
        plt.title(f"Distribution de puissance - Plan Z={z_level} - k-eff = {self.keff:.5f}")
        plt.tight_layout()
        plt.show()


# ===== SCRIPT PRINCIPAL =====
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(
        description="IAEA 3D - Solveur neutronique avancé"
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
    print("BENCHMARK IAEA 3D")
    print("="*60)
    print(f"Maillage : {args.mesh}")
    print(f"Domaine  : {args.domain}")
    print(f"Solveur  : {args.solver}")
    print(f"Upscatter: {'Oui' if args.upscatter else 'Non'}")
    print("="*60)
    
    # Création
    iaea3d = Iaea3D(meshtype=args.mesh, domaine=args.domain)
    iaea3d.order = args.ordre
    iaea3d.load_iaea3d_mat(include_upscattering=args.upscatter)
    iaea3d.mesh_initialisation()
    
    # Résolution avec le solveur spécifié
    iaea3d.init_solver()
    iaea3d.solve(solver_type=solver_map[args.solver])
    #iaea3d.phi_fins = iaea3d.mysolv.zoom((2,2,1))
    
    if args.adjoint_from_forward:
    	 # Problème adjoint
    	 iaea3d.solve(solver_type=solver_map[args.solver], adjoint=False, direct_adjoint=True)
    
    if args.adjoint:
    	 # Problème adjoint sans utiliser le forward
    	 iaea3d.solve(solver_type=solver_map[args.solver], adjoint=True, direct_adjoint=False)
    

    # Export
    iaea2d.mysolv.save_vtk("iaea3d_result")
    iaea2d.mysolv.save_vtk_coarse("iaea3d_result_coarse")
    iaea2d.mysolv.save_vtk_zoom("iaea3d_result_zoom")
    print("\n📁 Résultats exportés: iaea3d_result.vtk")
    
    # Visualisation
    if args.plot:
        iaea3d.plot_flux(group=0)
        iaea3d.plot_flux(group=1)
        iaea3d.plot_pvol()

        iaea3d.plot_flux(group=0, fine=True)
        iaea3d.plot_flux(group=1, fine=True)

    if args.harmonics > 0:
        # Calculer les harmoniques
        alphas = iaea3d.mysolv.solve_harmonics(n_harmonics=args.harmonics)
        print(f"Valeurs propres harmoniques: {alphas}")
        iaea3d.mysolv.save_harmonics_vtk("harmonics")

    print(f"\n⏱️  Temps total : {time.time() - iaea3d.start:.2f} s")
    print("="*60)
