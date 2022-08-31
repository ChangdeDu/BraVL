





class TBLogger():
    def __init__(self, name, writer):
        self.name = name;
        self.writer = writer;
        self.training_prefix = 'train';
        self.testing_prefix = 'test';
        self.step = 0;


    def write_log_probs(self, name, log_probs): self.writer.add_scalars('%s/LogProb' % name,
                                log_probs,
                                self.step)


    def write_klds(self, name, klds):
        self.writer.add_scalars('%s/KLD' % name,
                                klds,
                                self.step)


    def write_group_div(self, name, group_div):
        self.writer.add_scalars('%s/group_divergence' % name,
                                {'group_div': group_div.item()},
                                self.step)

    def write_latent_distr(self, name, latents):
        l_mods = latents['modalities'];
        for k, key in enumerate(l_mods.keys()):
            if not l_mods[key][0] is None:
                self.writer.add_scalars('%s/mu' % name,
                                        {key: l_mods[key][0].mean().item()},
                                        self.step)
            # if not l_mods[key][1] is None:
            #     self.writer.add_scalars('%s/logvar' % name,
            #                             {key: l_mods[key][1].mean().item()},
            #                             self.step)


    def write_lr_eval(self, lr_eval):
        for s, l_key in enumerate(sorted(lr_eval.keys())):
            self.writer.add_scalars('Latent Representation/%s'%(l_key),
                                    lr_eval[l_key],
                                    self.step)


    def write_coherence_logs(self, gen_eval):
        for j, l_key in enumerate(sorted(gen_eval['cond'].keys())):
            for k, s_key in enumerate(gen_eval['cond'][l_key].keys()):
                self.writer.add_scalars('Generation/%s/%s' %
                                        (l_key, s_key),
                                        gen_eval['cond'][l_key][s_key],
                                        self.step)
        self.writer.add_scalars('Generation/Random',
                                gen_eval['random'],
                                self.step)


    def write_lhood_logs(self, lhoods):
        for k, key in enumerate(sorted(lhoods.keys())):
            self.writer.add_scalars('Likelihoods/%s'%
                                    (key),
                                    lhoods[key],
                                    self.step)

    def write_prd_scores(self, prd_scores):
        self.writer.add_scalars('PRD',
                                prd_scores,
                                self.step)


    def write_plots(self, plots, epoch):
        for k, p_key in enumerate(plots.keys()):
            ps = plots[p_key];
            for l, name in enumerate(ps.keys()):
                fig = ps[name];
                self.writer.add_image(p_key + '_' + name,
                                      fig,
                                      epoch,
                                      dataformats="HWC");



    def add_basic_logs(self, name, results, loss, log_probs, klds,inter_mi):
        self.writer.add_scalars('%s/Loss' % name,
                                {'loss': loss.data.item()},
                                self.step)
        # self.writer.add_scalars( '%s/intra_mi' % name,
        #                         {'intra_mi': intra_mi},
        #                         self.step)
        self.writer.add_scalars( '%s/inter_mi' % name,
                                {'inter_mi': inter_mi},
                                self.step)
        self.write_log_probs(name, log_probs);
        self.write_klds(name, klds);
        self.write_group_div(name, results['joint_divergence']);
        self.write_latent_distr(name, results['latents']);


    def write_training_logs(self, results, loss, log_probs, klds,inter_mi):
        self.add_basic_logs(self.training_prefix, results, loss, log_probs, klds,inter_mi);
        self.step += 1;


    def write_testing_logs(self, results, loss, log_probs, klds,inter_mi):
        self.add_basic_logs(self.testing_prefix, results, loss, log_probs, klds,inter_mi);
        self.step += 1;





