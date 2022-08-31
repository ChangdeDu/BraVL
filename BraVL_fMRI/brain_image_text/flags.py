from utils.BaseFlags import parser as parser

# DATASET NAME
parser.add_argument('--dataset', type=str, default='Brain_Image_Text', help="name of the dataset")
# DATA DEPENDENT
# to be set by experiments themselves
parser.add_argument('--style_m1_dim', type=int, default=0, help="dimension of varying factor latent space")
parser.add_argument('--style_m2_dim', type=int, default=0, help="dimension of varying factor latent space")
parser.add_argument('--style_m3_dim', type=int, default=0, help="dimension of varying factor latent space")

parser.add_argument('--num_hidden_layers', type=int, default=2, help="number of channels in images")
parser.add_argument('--likelihood_m1', type=str, default='laplace', help="output distribution")
parser.add_argument('--likelihood_m2', type=str, default='laplace', help="output distribution")
parser.add_argument('--likelihood_m3', type=str, default='laplace', help="output distribution")

# LOSS TERM WEIGHTS
parser.add_argument('--beta_m1_style', type=float, default=1.0, help="default weight divergence term style modality 1")
parser.add_argument('--beta_m2_style', type=float, default=1.0, help="default weight divergence term style modality 2")
parser.add_argument('--beta_m3_style', type=float, default=1.0, help="default weight divergence term style modality 3")
parser.add_argument('--beta_m1_rec', type=float, default=1.0, help="default weight reconstruction modality 1")
parser.add_argument('--beta_m2_rec', type=float, default=1.0, help="default weight reconstruction modality 2")
parser.add_argument('--beta_m3_rec', type=float, default=1.0, help="default weight reconstruction modality 3")
parser.add_argument('--div_weight_m1_content', type=float, default=0.25, help="default weight divergence term content modality 1")
parser.add_argument('--div_weight_m2_content', type=float, default=0.25, help="default weight divergence term content modality 2")
parser.add_argument('--div_weight_m3_content', type=float, default=0.25, help="default weight divergence term content modality 2")
parser.add_argument('--div_weight_uniform_content', type=float, default=0.25, help="default weight divergence term prior")
#
