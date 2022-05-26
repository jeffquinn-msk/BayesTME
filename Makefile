wheels_osx11_m1:
	wget -Owheels_osx11_m1.tar.gz 'https://www.dropbox.com/s/ar44kpl3ri0k1gf/wheels_osx11_m1.tar.gz?dl=0'
	tar -xzvf wheels_osx11_m1.tar.gz

VE: wheels_osx11_m1
	pip3 install --upgrade pip	
	pip3 install --upgrade virtualenv
	python3 -m virtualenv VE
	VE/bin/pip3 install -r requirements.txt \
		--pre \
		--find-links wheels_osx11_m1 \
		--only-binary scipy,matplotlib,pypolyagamma,scikit_learn,scikit_image,scikit_sparse \
		--prefer-binary
	VE/bin/pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu
	VE/bin/python setup.py install
