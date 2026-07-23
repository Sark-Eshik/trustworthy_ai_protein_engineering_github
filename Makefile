.PHONY: env-conda env-pip

env-conda:
	@echo "Creating Conda environment 'trustworthy_ai_protein_engineering' from environment.yml..."
	conda env create -f environment.yml
	@echo ""
	@echo "========================================================================"
	@echo "Conda environment created successfully!"
	@echo "To activate it, run:"
	@echo "    conda activate trustworthy_ai_protein_engineering"
	@echo "========================================================================"

env-pip:
	@echo "Creating Python virtual environment in '.venv'..."
	python3 -m venv .venv
	@echo "Installing dependencies from requirements.txt into the virtual environment..."
	.venv/bin/pip install --upgrade pip
	.venv/bin/pip install -r requirements.txt
	@echo ""
	@echo "========================================================================"
	@echo "Virtual environment '.venv' created and dependencies installed successfully!"
	@echo "To activate it, run:"
	@echo "    source .venv/bin/activate"
	@echo "========================================================================"
