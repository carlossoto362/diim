ifeq ($(shell uname), Darwin)
        SHRC := $(HOME)/.zshrc
	SED_INPLACE := sed -i ''
else
        SHRC := $(HOME)/.bashrc
	SED_INPLACE := sed -i
endif

LINE := export DIIM_PATH=$(PWD)
PYTHON_VERSION := $(shell python3 -c "import sys; print(f'python{sys.version_info.major}.{sys.version_info.minor}')")

create_venv:
	@read -p "Enter the directory where you want to create the virtual environment (empty for current one): " ENVDIR; \
	if [ -z "$$ENVDIR" ]; then \
		ENVDIR=$(PWD); \
		echo "Using the current directory"; \
		$(PYTHON_VERSION) -m venv "$$ENVDIR/diim_env"; \
                echo "Virtual environment created in $$ENVDIR/diim_env"; \
	else \
		mkdir -p "$$ENVDIR"; \
		$(PYTHON_VERSION) -m venv "$$ENVDIR/diim_env"; \
		echo "Virtual environment created in $$ENVDIR/diim_env"; \
	fi; \
	make add_bashrc_line ENVDIR=$$ENVDIR

add_bashrc_line:
	grep -qxF '$(LINE)' $(SHRC) || echo $(LINE) >> $(SHRC)
	grep -qxF 'export DIIM_ENV_PATH=$(ENVDIR)' $(SHRC) || echo export DIIM_ENV_PATH=$(ENVDIR) >> $(SHRC)
	make setup ENVDIR=$(ENVDIR)

setup: requirements.txt
	echo $(ENVDIR)
	$(ENVDIR)/diim_env/bin/pip install networkx==2.8.8
	$(ENVDIR)/diim_env/bin/pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
	$(ENVDIR)/diim_env/bin/pip install -r requirements.txt
	cp -r ./diimpy $(ENVDIR)/diim_env/lib/$(PYTHON_VERSION)/site-packages/
	@echo run 'source $(ENVDIR)/diim_env/bin/activate' for activating diim env, 'deactivate' for deactivating it. 
	@echo also run 'source $(SHRC)' to let know to the scripts where is the home directory of DIIM. 

clean:
	@rm -f -r $(DIIM_ENV_PATH)/diim_env
	@$(SED_INPLACE) "\|$(LINE)|d" $(SHRC)
	@$(SED_INPLACE) "\|export DIIM_ENV_PATH=$(ENVDIR)|d" $(SHRC)

