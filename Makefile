# =========================================================================
# DÉFINITION DES CHEMINS DES PRÉREQUIS
# =========================================================================

# GCC, pybind11 et python
#GCC = /home/julien/Bureau/HADES/prerequis/gcc-13.2.0/install
#ANACONDA_VERSION = /home/julien/anaconda3/envs/mfem
#PYBIND = /home/julien/Bureau/HADES/prerequis/pybind11-2.13.6
GCC = XXXX TO COMPLETE XXXX
ANACONDA_VERSION = XXXX TO COMPLETE XXXX
PYBIND = XXXX TO COMPLETE XXXX

# =========================================================================
# COMPILATEUR ET DRAPEAUX
# =========================================================================

CXX = $(GCC)/bin/g++-13.2.0

# -fPIC (position independent code) est essentiel pour les bibliothèques partagées (.so)
CXXFLAGS = -fPIC -O3 -std=c++17 -march=native -Wno-deprecated -fvisibility=hidden

# Nom du module de sortie
COXLIB = neutmfem/_neutmfem.so

# =========================================================================
# INCLUSIONS ET LIENS
# =========================================================================

# Chemins d'en-têtes (pour la compilation des .o)
INC = -I $(ANACONDA_VERSION)/include/mfem -I $(ANACONDA_VERSION)/include -I $(ANACONDA_VERSION)/include/python3.13/ \
      -I $(PYBIND)/include -I ./include

# Dépendances systèmes critiques pour la liaison partagée
SYS_LIBS = -lrt -lstdc++ -lm -ldl -lpthread

# Chemins des bibliothèques (-L):
LDFLAGS = -L $(ANACONDA_VERSION)/lib

# Bibliothèques à lier (-l)
LDLIBS = -lmfem $(SYS_LIBS)

# Fichiers sources (objets intermédiaires)
SRC = lib/NeutMFEM.o lib/wrapper.o lib/solver.o

# Répertoires à créer
DIRS = lib

# =========================================================================
# RÈGLES MAKE
# =========================================================================

# Règle pour créer les répertoires
$(DIRS):
	mkdir -p $@

# Règle principale: dépend de la création des dossiers et du fichier final
all : $(DIRS) $(COXLIB)

# Règle de liaison: dépend des objets. CORRECTION: $^ ne contient plus $(DIRS)
$(COXLIB): $(SRC)
	@echo "Linking shared library $(COXLIB)..."
	$(CXX) -shared -fopenmp -o $@ $^ $(LDFLAGS) $(LDLIBS)

# Règle de compilation des objets (.o)
lib/%.o : src/%.cpp
	@echo "Compiling $<..."
	$(CXX) -c $< -o $@ $(INC) -fopenmp $(CXXFLAGS)

clean :
	rm -f lib/*.o neutmfem/*.so
