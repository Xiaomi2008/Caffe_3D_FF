classdef CaffeNet < handle
    properties (SetAccess = private)
        %model_def_file = '../../examples/imagenet/imagenet_deploy.prototxt';
        %root_dir ='/home/tzeng/ABA/autoGenelable_multi_lables_proj/code/caffe/models/';
		%model_def_file = [root_dir ''];
		%model_def_file='/home/tzeng/ABA/autoGenelable_multi_lables_proj/code/caffe/models/vgg_16/deploy.prototxt';
		%model_def_file='/home/tzeng/ABA/autoGenelable_multi_lables_proj/code/caffe/models/vgg_16_fine_tune_gen/deploy.prototxt';
        %model_file = '../../examples/imagenet/caffe_reference_imagenet_model';
		%model_file = '/home/tzeng/ABA/autoGenelable_multi_lables_proj/code/caffe/models/vgg_16/VGG_16.caffemodel';
		%model_file =  '/home/tzeng/ABA/autoGenelable_multi_lables_proj/code/caffe/models/vgg_16_fine_tune_gen/finetune_vgg_16_brain_onotology_3color_mean_iter_5000';
		%model_file ='/home/tzeng/ABA/autoGenelable_multi_lables_proj/code/caffe/models/finetune_vgg_16_brain_onotology_3color_mean_noCrop_last_L_3-6_L0.001_step2500_iter_12000';
		
		model_def_file ='/home/tzeng/ABA/RL_gene_ontology_proj/caffe/models/vgg_16_fine_tune/deploy.prototxt';
		model_file    ='/home/tzeng/ABA/RL_gene_ontology_proj/caffe/models/finetune_vgg_16_100_labels_GO_lR_E-05_Step-2000_msk_iter_10000';
		
		
		
        layers_info
        blobs_info
    end
    properties
        mode
        phase
        device_id
        input_blobs
        output_blobs
    end
    properties (Dependent)
        weights
    end
    properties (Access = private)
        init_key
        weights_changed = true;
        weights_store
    end
    methods (Access=private)
        function self = CaffeNet(model_def_file, model_file)
            if nargin == 0
                init(self, self.model_def_file, self.model_file);
            end
            if nargin > 0
                init_net(self, model_def_file);
            end
            if nargin > 1
                load_net(self, model_file);
            end
            assert(is_initialized(self))
            self.mode = caffe_safe('get_mode');
            self.phase = caffe_safe('get_phase');
            self.device_id = caffe_safe('get_device');
        end
    end
    methods (Static)
        function obj = instance(model_def_file, model_file)
            persistent self
            if isempty(self)
                switch nargin
                    case 2
                        self = CaffeNet(model_def_file, model_file);
                    case 1
                        self = CaffeNet(model_def_file);
                    case 0
                        self = CaffeNet();
                end
            else
                if nargin > 0 && ~isempty(model_def_file)
                    init_net(self,model_def_file);
                    disp(' init_net(self,model_def_file)');
                end
                if nargin > 1 && ~isempty(model_file)
                    load_net(self,model_file);
                    disp('load_net(self,model_file)');
                end
            end
            obj = self;
        end
    end
    methods
        function weights = get.weights(self)
            assert(is_initialized(self))
            if (self.weights_changed)
                self.weights_store = caffe_safe('get_weights');
                self.weights_changed = false;
            end
            weights = self.weights_store;
        end
        function set.weights(self,weights)
            assert(is_initialized(self))
            caffe_safe('set_weights', weights);
            self.weights_store = weights;
            self.weights_changed = false;
        end
        function set.mode(self,mode)
            % mode = {'CPU' 'GPU'}
            switch mode
                case 'CPU'
                    caffe_safe('set_mode_cpu');
                    self.mode = mode;
                case 'GPU'
                    caffe_safe('set_mode_gpu');
                    self.mode = mode;
                otherwise
                    fprintf('Mode unknown choose between CPU and GPU\n');
                    error('Mode unknown');
            end
        end
        function set.phase(self, phase)
            % phase = {'TRAIN' 'TEST'}
            switch phase
                case 'TRAIN'
                    caffe_safe('set_phase_train');
                    self.phase = phase;
                case {'test','TEST'}
                    caffe_safe('set_phase_test');
                    self.phase = phase;
                otherwise
                    fprintf('Phase unknown choose between TRAIN and TEST')
                    error('Phase unknown');
            end
        end
        function set.device_id(self, device_id)
            caffe_safe('set_device', device_id);
            self.device_id = device_id;
        end
        function set.input_blobs(self, input_blobs)
            assert(is_initialized(self))
            caffe_safe('set_input_blobs', input_blobs);
            self.input_blobs = input_blobs;
        end
        function set.output_blobs(self, output_blobs)
            assert(is_initialized(self))
            caffe_safe('set_output_blobs', output_blobs);
            self.output_blobs = output_blobs;
        end
    end
    methods
        function res = forward(self,input)
            assert(is_initialized(self))
            if nargin < 2
                res = caffe_safe('forward');
            else
                res = caffe_safe('forward',input);
            end
        end
        function res = backward(self,diff)
            assert(is_initialized(self))
            if nargin < 2
                res = caffe_safe('backward');
            else
                res = caffe_safe('backward',diff);
            end
        end
        function res = forward_prefilled(self)
            assert(is_initialized(self))
            res = caffe_safe('forward_prefilled');
        end
        function res = backward_prefilled(self)
            assert(is_initialized(self))
            res = caffe_safe('backward_prefilled');
        end
        function res = init(self, model_def_file, model_file)
            self.init_key = caffe_safe('init',model_def_file, model_file);
            assert(is_initialized(self))
            self.model_def_file = model_def_file;
            self.model_file = model_file;
            self.layers_info = caffe_safe('get_layers_info');
            self.blobs_info = caffe_safe('get_blobs_info');
            self.weights_changed = true;
            res = self.init_key;
        end
        function res = init_net(self, model_def_file)
            self.init_key = caffe_safe('init_net',model_def_file);
            assert(is_initialized(self))
            self.model_def_file = model_def_file;
            self.model_file = [];
            self.layers_info = caffe_safe('get_layers_info');
            self.blobs_info = caffe_safe('get_blobs_info');
            self.weights_changed = true;
            res = self.init_key;
        end
        function res = load_net(self, model_file)
            assert(is_initialized(self))
            self.init_key = caffe_safe('load_net',model_file);
            self.model_file = model_file;
            self.weights_changed = true;
            res = self.init_key;
        end
        function res = save_net(self, model_file)
            assert(is_initialized(self))
            res = caffe_safe('save_net', model_file);
        end
        function res = is_initialized(~)
            res = caffe_safe('is_initialized')>0;
        end
        function set_mode_cpu(self)
            self.mode = 'CPU';
        end
        function set_mode_gpu(self)
            self.mode = 'GPU';
        end
        function set_phase_train(self)
            self.phase = 'TRAIN';
        end
        function set_phase_test(self)
            self.phase = 'TEST';
        end
        function set_device(self, device_id)
            self.device_id = device_id;
        end
        function res = get_weights(self)
            assert(is_initialized(self))
            res = self.weights;
        end
        function set_weights(self, weights)
            self.weights = weights;
        end
        function res = get_layer_weights(self, layer_name)
            assert(is_initialized(self))
            res = caffe_safe('get_layer_weights', layer_name);
        end
        function res = set_layer_weights(self, layer_name, weights)
            assert(is_initialized(self))
            res = caffe_safe('set_layer_weights', layer_name, weights);
            self.weights = caffe_safe('get_weights');
        end
        function res = get_layers_info(self)
            assert(is_initialized(self))
            res = self.layers_info;
        end
        function res = get_blobs_info(self)
            assert(is_initialized(self))
            res = self.blobs_info;
        end
        function res = get_blob_data(self, blob_name)
            assert(is_initialized(self))
            res = caffe_safe('get_blob_data', blob_name);
        end
        function res = get_blob_diff(self, blob_name)
            assert(is_initialized(self))
            res = caffe_safe('get_blob_diff', blob_name);
        end
        function res = get_all_data(self)
            assert(is_initialized(self))
            res = caffe_safe('get_all_data');
        end
        function res = get_all_diff(self)
            assert(is_initialized(self))
            res = caffe_safe('get_all_diff');
        end
        function res = get_init_key(self)
            self.init_key = caffe_safe('get_init_key');
            res = self.init_key;
        end
        function reset(self)
            caffe_safe('reset');
            self.init_key = caffe_safe('get_init_key');
            self.layers_info = [];
            self.blobs_info = [];
            self.weights_store = [];
        end
        function delete(self)
            self.weights_store = [];
            caffe_safe('reset');
            clear caffe;
        end
    end
end
