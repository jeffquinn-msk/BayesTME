VE:
	pip3 install --upgrade pip	
	pip3 install --upgrade virtualenv
	python3 -m virtualenv VE
	VE/bin/pip3 install -r requirements.txt 
	VE/bin/pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu
