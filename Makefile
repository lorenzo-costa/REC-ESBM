# Makefile

# Variables
PYTHON=python
SRC=src
DATA=data
RESULTS=results

# Default target
all: preprocess simulate analyze figures

# Step 1: Preprocess data
preprocess:
	$(PYTHON) $(SRC)/pipeline/pre_processing_functs.py

# Step 2: Run simulations
simulate:
	$(PYTHON) $(SRC)/analysis/simulations.py

# Step 3: Analyze book data
analyze:
	$(PYTHON) $(SRC)/analysis/book_analysis.py

# Step 4: Generate figures (optional if scripts already output plots)
figures:
	@echo "Figures should now be in $(RESULTS)/figures"

# Clean up caches
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
