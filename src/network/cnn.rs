use burn::nn::conv::{Conv2d, Conv2dConfig};
use burn::nn::PaddingConfig2d;
use burn::prelude::*;
use burn::tensor::activation::relu;

use crate::config::Config;
use crate::profile::{profile_function, profile_scope};

use super::mlp::create_linear_orthogonal;

/// CNN Actor-Critic network for spatial observations
///
/// Architecture:
/// 1. Conv layers process spatial features (stride=1, same padding)
/// 2. Flatten conv output
/// 3. Concatenate extra features (e.g., turn indicator)
/// 4. FC layers process combined features
/// 5. Policy and value heads
///
/// When `split_networks` is true: separate conv+FC stacks for actor and critic
///
/// The value head outputs a single scalar value (the acting player's value).
#[derive(Module, Debug)]
pub struct CnnActorCritic<B: Backend> {
    /// Actor convolutional layers
    pub conv_layers: Vec<Conv2d<B>>,
    /// Actor FC layers after flattening
    pub fc_layers: Vec<nn::Linear<B>>,
    /// Critic convolutional layers (only populated when `split_networks` is true)
    pub critic_conv_layers: Vec<Conv2d<B>>,
    /// Critic FC layers (only populated when `split_networks` is true)
    pub critic_fc_layers: Vec<nn::Linear<B>>,
    /// Policy output head
    pub policy_head: nn::Linear<B>,
    /// Single value head - outputs scalar (acting player's value)
    pub value_head: nn::Linear<B>,
    /// Spatial input shape (H, W, C)
    #[module(skip)]
    input_shape: (usize, usize, usize),
    /// Number of extra features (concatenated after conv)
    #[module(skip)]
    extra_features_size: usize,
    /// Use `ReLU` activation (always true for CNN, but kept for consistency)
    #[module(skip)]
    use_relu: bool,
    /// Whether to use separate actor/critic networks
    #[module(skip)]
    split_networks: bool,
}

impl<B: Backend> CnnActorCritic<B> {
    /// Create a new CNN Actor-Critic network
    ///
    /// # Arguments
    /// * `obs_dim` - Total observation dimension (spatial + extra features)
    /// * `obs_shape` - Spatial shape (height, width, channels)
    /// * `action_count` - Number of discrete actions
    /// * `config` - Configuration with CNN parameters
    /// * `device` - Compute device
    ///
    /// # Panics
    /// Panics if `obs_shape` is None (CNN requires spatial observations)
    ///
    /// The value head outputs a single scalar (the acting player's value).
    pub fn new(
        obs_dim: usize,
        obs_shape: Option<(usize, usize, usize)>,
        action_count: usize,
        config: &Config,
        device: &B::Device,
    ) -> Self {
        let (height, width, channels) =
            obs_shape.expect("CNN requires OBSERVATION_SHAPE to be set");
        let spatial_size = height * width * channels;
        let extra_features_size = obs_dim - spatial_size;

        let num_conv_layers = config.num_conv_layers;
        let kernel_size = config.kernel_size;
        let fc_hidden_size = config.cnn_fc_hidden_size;
        let num_fc_layers = config.cnn_num_fc_layers;
        let use_relu = config.activation == "relu";
        let split_networks = config.split_networks;

        // Get channel counts, repeating last value if needed
        let get_channels = |layer_idx: usize| -> usize {
            if layer_idx < config.conv_channels.len() {
                config.conv_channels[layer_idx]
            } else {
                *config.conv_channels.last().unwrap_or(&64)
            }
        };

        // Build actor conv layers
        let conv_layers = Self::build_conv_layers(
            channels,
            num_conv_layers,
            kernel_size,
            &get_channels,
            device,
        );

        // Calculate flattened conv output size (spatial preserved with same padding)
        let final_channels = get_channels(num_conv_layers.saturating_sub(1));
        let conv_output_size = height * width * final_channels;
        let fc_input_size = conv_output_size + extra_features_size;

        // Build actor FC layers
        let fc_layers = Self::build_fc_layers(
            fc_input_size,
            fc_hidden_size,
            num_fc_layers,
            use_relu,
            device,
        );

        // Build critic networks if using split architecture
        let (critic_conv_layers, critic_fc_layers) = if split_networks {
            let critic_conv = Self::build_conv_layers(
                channels,
                num_conv_layers,
                kernel_size,
                &get_channels,
                device,
            );
            let critic_fc = Self::build_fc_layers(
                fc_input_size,
                fc_hidden_size,
                num_fc_layers,
                use_relu,
                device,
            );
            (critic_conv, critic_fc)
        } else {
            (Vec::new(), Vec::new())
        };

        // Policy head: small init (0.01) for stable initial policy
        let policy_head = create_linear_orthogonal(fc_hidden_size, action_count, 0.01, device);

        // Value head: single output (acting player's value), gain 1.0
        let value_head = create_linear_orthogonal(fc_hidden_size, 1, 1.0, device);

        Self {
            conv_layers,
            fc_layers,
            critic_conv_layers,
            critic_fc_layers,
            policy_head,
            value_head,
            input_shape: (height, width, channels),
            extra_features_size,
            use_relu,
            split_networks,
        }
    }

    /// Build convolutional layers
    fn build_conv_layers<F>(
        in_channels: usize,
        num_layers: usize,
        kernel_size: usize,
        get_channels: &F,
        device: &B::Device,
    ) -> Vec<Conv2d<B>>
    where
        F: Fn(usize) -> usize,
    {
        let mut layers = Vec::with_capacity(num_layers);
        let mut current_channels = in_channels;

        for i in 0..num_layers {
            let out_channels = get_channels(i);
            let config =
                Conv2dConfig::new([current_channels, out_channels], [kernel_size, kernel_size])
                    .with_stride([1, 1])
                    .with_padding(PaddingConfig2d::Same);
            layers.push(config.init(device));
            current_channels = out_channels;
        }

        layers
    }

    /// Build fully-connected layers
    fn build_fc_layers(
        input_size: usize,
        hidden_size: usize,
        num_layers: usize,
        use_relu: bool,
        device: &B::Device,
    ) -> Vec<nn::Linear<B>> {
        let hidden_gain = if use_relu { 2.0_f64.sqrt() } else { 1.0 };
        let mut layers = Vec::with_capacity(num_layers);
        let mut in_size = input_size;

        for _ in 0..num_layers {
            layers.push(create_linear_orthogonal(
                in_size,
                hidden_size,
                hidden_gain,
                device,
            ));
            in_size = hidden_size;
        }

        layers
    }

    /// Forward pass for conv layers
    fn forward_conv(spatial: Tensor<B, 4>, conv_layers: &[Conv2d<B>]) -> Tensor<B, 2> {
        let mut x = spatial;
        for layer in conv_layers {
            x = layer.forward(x);
            x = relu(x);
        }
        // Flatten: [batch, C, H, W] -> [batch, C*H*W]
        let [batch, c, h, w] = x.dims();
        x.reshape([batch, c * h * w])
    }

    /// Forward pass for FC layers
    fn forward_fc(&self, x: Tensor<B, 2>, fc_layers: &[nn::Linear<B>]) -> Tensor<B, 2> {
        let mut x = x;
        for layer in fc_layers {
            x = layer.forward(x);
            x = if self.use_relu { relu(x) } else { x.tanh() };
        }
        x
    }

    /// Forward pass returning action logits and value
    ///
    /// Input: observations [batch, `obs_dim`] (flat, will be reshaped internally)
    /// Output: (logits [batch, `action_count`], values [batch, 1])
    pub fn forward(&self, obs: Tensor<B, 2>) -> (Tensor<B, 2>, Tensor<B, 2>) {
        profile_function!();

        let (height, width, channels) = self.input_shape;
        let spatial_size = height * width * channels;
        let [batch, obs_dim] = obs.dims();

        // Split observation into spatial and extra features
        let (spatial_flat, extra) = {
            profile_scope!("split_obs");
            let spatial_flat = obs.clone().slice([0..batch, 0..spatial_size]);
            let extra = if self.extra_features_size > 0 {
                Some(obs.slice([0..batch, spatial_size..obs_dim]))
            } else {
                None
            };
            (spatial_flat, extra)
        };

        // Reshape spatial to [batch, C, H, W] (NCHW format)
        let spatial = {
            profile_scope!("reshape_spatial");
            // Input is [batch, H*W*C] in row-major order with channels last
            // Need to reshape to [batch, H, W, C] then permute to [batch, C, H, W]
            spatial_flat
                .reshape([batch, height, width, channels])
                .permute([0, 3, 1, 2]) // [batch, C, H, W]
        };

        if self.split_networks {
            // Separate actor and critic networks
            let actor_features = {
                profile_scope!("actor_conv");
                Self::forward_conv(spatial.clone(), &self.conv_layers)
            };
            let actor_combined = if let Some(ref extra) = extra {
                Tensor::cat(vec![actor_features, extra.clone()], 1)
            } else {
                actor_features
            };
            let actor_out = {
                profile_scope!("actor_fc");
                self.forward_fc(actor_combined, &self.fc_layers)
            };
            let logits = {
                profile_scope!("policy_head");
                self.policy_head.forward(actor_out)
            };

            let critic_features = {
                profile_scope!("critic_conv");
                Self::forward_conv(spatial, &self.critic_conv_layers)
            };
            let critic_combined = if let Some(extra) = extra {
                Tensor::cat(vec![critic_features, extra], 1)
            } else {
                critic_features
            };
            let critic_out = {
                profile_scope!("critic_fc");
                self.forward_fc(critic_combined, &self.critic_fc_layers)
            };
            let values = {
                profile_scope!("value_head");
                self.value_head.forward(critic_out)
            };

            (logits, values)
        } else {
            // Shared backbone with separate heads
            let features = {
                profile_scope!("shared_conv");
                Self::forward_conv(spatial, &self.conv_layers)
            };
            let combined = if let Some(extra) = extra {
                Tensor::cat(vec![features, extra], 1)
            } else {
                features
            };
            let out = {
                profile_scope!("shared_fc");
                self.forward_fc(combined, &self.fc_layers)
            };

            let logits = {
                profile_scope!("policy_head");
                self.policy_head.forward(out.clone())
            };
            let values = {
                profile_scope!("value_head");
                self.value_head.forward(out)
            };

            (logits, values)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;

    type TestBackend = NdArray<f32>;

    fn cnn_config() -> Config {
        Config {
            network_type: "cnn".to_string(),
            num_conv_layers: 2,
            conv_channels: vec![32, 64],
            kernel_size: 3,
            cnn_fc_hidden_size: 64,
            cnn_num_fc_layers: 1,
            ..Config::default()
        }
    }

    #[test]
    fn test_cnn_forward_shape() {
        let device = Default::default();
        let config = cnn_config();

        // Connect Four: 86 obs_dim, (6, 7, 2) shape, 7 actions
        let model: CnnActorCritic<TestBackend> =
            CnnActorCritic::new(86, Some((6, 7, 2)), 7, &config, &device);

        let batch_size = 8;
        let obs = Tensor::zeros([batch_size, 86], &device);
        let (logits, values) = model.forward(obs);

        assert_eq!(logits.dims(), [batch_size, 7]);
        assert_eq!(values.dims(), [batch_size, 1]); // Single scalar value
    }

    #[test]
    fn test_cnn_forward_no_extra_features() {
        let device = Default::default();
        let config = cnn_config();

        // 4x4x2 = 32 spatial, no extra features
        let model: CnnActorCritic<TestBackend> =
            CnnActorCritic::new(32, Some((4, 4, 2)), 4, &config, &device);

        let batch_size = 4;
        let obs = Tensor::zeros([batch_size, 32], &device);
        let (logits, values) = model.forward(obs);

        assert_eq!(logits.dims(), [batch_size, 4]);
        assert_eq!(values.dims(), [batch_size, 1]);
    }

    #[test]
    fn test_cnn_split_networks() {
        let device = Default::default();
        let config = Config {
            split_networks: true,
            ..cnn_config()
        };

        let model: CnnActorCritic<TestBackend> =
            CnnActorCritic::new(86, Some((6, 7, 2)), 7, &config, &device);

        // Verify split structure
        assert!(!model.critic_conv_layers.is_empty());
        assert!(!model.critic_fc_layers.is_empty());
        assert_eq!(model.critic_conv_layers.len(), config.num_conv_layers);

        let batch_size = 4;
        let obs = Tensor::zeros([batch_size, 86], &device);
        let (logits, values) = model.forward(obs);

        assert_eq!(logits.dims(), [batch_size, 7]);
        assert_eq!(values.dims(), [batch_size, 1]);
    }

    #[test]
    fn test_cnn_spatial_preserved() {
        // With stride=1 and same padding, spatial dimensions should be preserved
        // Note: Burn requires odd kernel sizes for "same" padding
        let device = Default::default();
        let config = Config {
            kernel_size: 3, // 3x3 kernel (must be odd for Burn's Same padding)
            ..cnn_config()
        };

        let model: CnnActorCritic<TestBackend> =
            CnnActorCritic::new(86, Some((6, 7, 2)), 7, &config, &device);

        // Just verify forward pass works (spatial dims preserved internally)
        let batch_size = 2;
        let obs = Tensor::ones([batch_size, 86], &device);
        let (logits, values) = model.forward(obs);

        assert_eq!(logits.dims(), [batch_size, 7]);
        assert_eq!(values.dims(), [batch_size, 1]);

        // Values should be finite
        let value_data = values.into_data();
        for v in value_data.as_slice::<f32>().unwrap() {
            assert!(v.is_finite());
        }
    }

    #[test]
    #[should_panic(expected = "CNN requires OBSERVATION_SHAPE")]
    fn test_cnn_requires_obs_shape() {
        let device = Default::default();
        let config = cnn_config();

        // This should panic because obs_shape is None
        let _model: CnnActorCritic<TestBackend> =
            CnnActorCritic::new(86, None, 7, &config, &device);
    }
}
