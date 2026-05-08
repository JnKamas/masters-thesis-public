import torch
import torch.nn as nn

import torchvision
from blitz.modules import BayesianLinear
from blitz.utils import variational_estimator

@variational_estimator
class Network(nn.Module):
    def __init__(self, args):
        super().__init__()

        # Dropout probabilities
        self.p_backbone = getattr(args, "dropout_prob_backbone", 0.0)
        self.p_rot = getattr(args, "dropout_prob_rot", 0.0)
        self.p_trans = getattr(args, "dropout_prob_trans", 0.0)

        self.use_aleatoric = getattr(args, "use_aleatoric", False)

        if args.backbone == 'resnet18':
            pretrained_backbone_model  = torchvision.models.resnet18(pretrained=True)
        elif args.backbone == 'resnet34':
            pretrained_backbone_model  = torchvision.models.resnet34(pretrained=True)
        elif args.backbone == 'resnet50':
            pretrained_backbone_model  = torchvision.models.resnet50(pretrained=True)
        else:
            raise ValueError(f"Unsupported backbone: {args.backbone}")

        last_feat = list(pretrained_backbone_model.children())[-1].in_features // 2
        self.backbone = nn.Sequential(*list(pretrained_backbone_model.children())[:-3])

        # Heads 
        def make_head(input_feat, output_feat):
            return torch.nn.Sequential(
                torch.nn.Linear(input_feat, 128),
                torch.nn.LeakyReLU(),
                torch.nn.Linear(128, 64),
                torch.nn.LeakyReLU(),
                torch.nn.Linear(64, output_feat)
            )

        def make_dropout_head(input_feat, output_feat, p):
            return torch.nn.Sequential(
                torch.nn.Linear(input_feat, 128),
                torch.nn.LeakyReLU(),
                torch.nn.Dropout(p),
                torch.nn.Linear(128, 64),
                torch.nn.LeakyReLU(),
                torch.nn.Dropout(p),
                torch.nn.Linear(64, output_feat)
            )

        def make_bayesian_head(input_feat, output_feat, btype):
            if btype == 1:
                return nn.Sequential(
                    BayesianLinear(input_feat, 128),
                    nn.LeakyReLU(),
                    nn.Linear(128, 64),
                    nn.LeakyReLU(),
                    nn.Linear(64, output_feat),
                )
            if btype == 2:
                return nn.Sequential(
                    nn.Linear(input_feat, 128),
                    nn.LeakyReLU(),
                    nn.Linear(128, 64),
                    nn.LeakyReLU(),
                    BayesianLinear(64, output_feat),
                )
            if btype == 3:
                return nn.Sequential(
                    nn.Linear(input_feat, 128),
                    nn.LeakyReLU(),
                    BayesianLinear(128, 64),
                    nn.LeakyReLU(),
                    nn.Linear(64, output_feat),
                )
            return nn.Sequential(
                BayesianLinear(input_feat, 128),
                nn.LeakyReLU(),
                BayesianLinear(128, 64),
                nn.LeakyReLU(),
                BayesianLinear(64, output_feat),
            )
        
        # dimension is extended to four when using aleatoric uncertainty
        output_feat_rot = 4 if self.use_aleatoric else 3 # only for one rotation vector.
        outpot_feat_trans = 6 if self.use_aleatoric else 3 # we dont predict just pose but also std
        if args.modifications == "mc_dropout":
            self.fc_z = make_dropout_head(last_feat, 3, self.p_rot)
            self.fc_y = make_dropout_head(last_feat, output_feat_rot, self.p_rot)
            self.fc_t = make_dropout_head(last_feat, outpot_feat_trans, self.p_trans)
        elif args.modifications == "bayesian":
            self.fc_z = make_bayesian_head(last_feat, 3, args.bayesian_type)
            self.fc_y = make_bayesian_head(last_feat, output_feat_rot, args.bayesian_type)
            self.fc_t = make_bayesian_head(last_feat, outpot_feat_trans, args.bayesian_type)
        else:
            self.fc_z = make_head(last_feat, 3)
            self.fc_y = make_head(last_feat, output_feat_rot)
            self.fc_t = make_head(last_feat, outpot_feat_trans)
        self.post_dropout = nn.Dropout(self.p_backbone)
    

    # ------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------
    def forward(self, x):
        x = self.backbone(x)

        # Global average pooling
        x = torch.mean(x, dim=-1)
        x = torch.mean(x, dim=-1)

        x = self.post_dropout(x)

        z = self.fc_z(x)
        y = self.fc_y(x)
        t = self.fc_t(x)

        if self.use_aleatoric:
            y_vec = y[:, :3]
            sigma_r = y[:, 3:4]

            t_vec = t[:, :3]
            s_t = t[:, 3:]

            return z, y_vec, t_vec, sigma_r, s_t

        return z, y, t