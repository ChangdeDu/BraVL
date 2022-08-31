from __future__ import print_function
from itertools import product
import os
import pickle
import bdpy
from bdpy.dataform import Features
from bdpy.util import dump_info, makedir_ifnot
import numpy as np
from sklearn.decomposition import PCA
from scipy import io

# Settings ###################################################################
seed = 42
TINY = 1e-8
# Python RNG
np.random.seed(seed)

subject_set=['subject1','subject2','subject3','subject4','subject5']
for subject in subject_set:
    if subject == 'subject1':
        subjects_list = {
            'sub-01':  'Subject1.h5',
        }

        subjects_list_test = {
            'sub-01':  'Subject1.h5',
        }
    elif subject == 'subject2':
        subjects_list = {
            'sub-02':  'Subject2.h5',
        }

        subjects_list_test = {
            'sub-02':  'Subject2.h5',
        }
    elif subject == 'subject3':
        subjects_list = {
            'sub-03':  'Subject3.h5',
        }

        subjects_list_test = {
            'sub-03':  'Subject3.h5',
        }
    elif subject == 'subject4':
        subjects_list = {
            'sub-04':  'Subject4.h5',
        }

        subjects_list_test = {
            'sub-04':  'Subject4.h5',
        }
    elif subject == 'subject5':
        subjects_list = {
            'sub-05':  'Subject5.h5',
        }

        subjects_list_test = {
            'sub-05':  'Subject5.h5',
        }

    text_embedding_list = [
                            'GPTNeo',
                            'ALBERT',
                            # 'GPTNeo_phrases',
                            # 'ALBERT_phrases'
    ]


    rois_list = {
        # 'VC':  'ROI_VC = 1',
        # 'LVC': 'ROI_LVC = 1',
        # 'HVC': 'ROI_HVC = 1',
        'V1':  'ROI_V1 = 1',
        'V2':  'ROI_V2 = 1',
        'V3':  'ROI_V3 = 1',
        'V4':  'ROI_V4 = 1',
        'LOC': 'ROI_LOC = 1',
        'FFA': 'ROI_FFA = 1',
        'PPA': 'ROI_PPA = 1',
    }

    network = 'pytorch/repvgg_b3g4'
    features_list = [  # 'Conv_0',
        # 'Conv_1',
        'Conv_2',
        'Conv_3',
        'Conv_4',
        'linear',
        'final']

    features_list = features_list[::-1]  # Start training from deep layers

    # Brain data
    brain_dir = './data/GenericObjectDecoding-v2'
    # Image features
    timm_extracted_visual_features = './data/GOD-Wiki/visual_feature/ImageNetTraining/' + network
    timm_extracted_visual_features_test = './data/GOD-Wiki/visual_feature/ImageNetTest/' + network
    timm_extracted_visual_features_aug = './data/GOD-Wiki/visual_feature/Aug_1000/' + network
    print('DNN feature')
    print(timm_extracted_visual_features)
    # Text features
    model_extracted_textual_features = './data/Wiki_articles_features'

    # Results directory
    results_dir_root = './data/GOD-Wiki/visual_feature/ImageNetTraining/' + network + '-PCA'
    results_dir_root_test = './data/GOD-Wiki/visual_feature/ImageNetTest/' + network + '-PCA'
    results_dir_root_aug = './data/GOD-Wiki/visual_feature/Aug_1000/' + network + '-PCA'
    results_fmri_root = './data/GOD-Wiki/brain_feature/LVC_HVC_IT'
    results_text_root = './data/GOD-Wiki/textual_feature/ImageNetTraining/text'
    results_text_root_test = './data/GOD-Wiki/textual_feature/ImageNetTest/text'
    results_text_root_aug = './data/GOD-Wiki/textual_feature/Aug_1000/text'

    # Main #######################################################################
    analysis_basename = os.path.splitext(os.path.basename(__file__))[0]
    # Print info -----------------------------------------------------------------
    print('Subjects:        %s' % subjects_list.keys())
    print('ROIs:            %s' % rois_list.keys())
    print('Target features: %s' % network.split('/')[-1])
    print('Layers:          %s' % features_list)
    print('')

    # Load data ------------------------------------------------------------------
    print('----------------------------------------')
    print('Loading data')

    data_brain = {sbj: bdpy.BData(os.path.join(brain_dir, dat_file))
                  for sbj, dat_file in subjects_list.items()}
    data_features = Features(os.path.join(timm_extracted_visual_features, network))

    data_brain_test = {sbj: bdpy.BData(os.path.join(brain_dir, dat_file))
                  for sbj, dat_file in subjects_list_test.items()}
    data_features_test = Features(os.path.join(timm_extracted_visual_features_test, network))
    data_features_aug = Features(os.path.join(timm_extracted_visual_features_aug, network))

    # Initialize directories -----------------------------------------------------
    makedir_ifnot(results_dir_root)
    makedir_ifnot(results_dir_root_test)
    makedir_ifnot(results_dir_root_aug)
    makedir_ifnot(results_text_root)
    makedir_ifnot(results_text_root_test)
    makedir_ifnot(results_text_root_aug)

    # Save runtime information ---------------------------------------------------
    info_dir = results_dir_root
    runtime_params = {
        'fMRI data':                [os.path.abspath(os.path.join(brain_dir, v)) for v in subjects_list.values()],
        'ROIs':                     rois_list.keys(),
        'target DNN':               network.split('/')[-1],
        'target DNN features':      os.path.abspath(timm_extracted_visual_features),
        'target DNN layers':        features_list,
    }
    dump_info(info_dir, script=__file__, parameters=runtime_params)


    #######################################
    # Original
    #######################################
    first = 1
    for sbj, roi in product(subjects_list ,rois_list):
        print('--------------------')
        print('VC ROI:        %s' % roi)
        # Brain data
        # data_brain[sbj].show_metadata()
        x= data_brain[sbj].select(rois_list[roi])
        x_labels = data_brain[sbj].select('image_index').flatten()  # Label (image index)
        print('roi_shape=', x.shape)

        x_test= data_brain_test[sbj].select(rois_list[roi])          # Brain data
        x_labels_test = data_brain_test[sbj].select('image_index').flatten()  # Label (image index)

        if first:
            best_roi_sel = x
            best_roi_sel_test = x_test
            first = 0
        else:
            best_roi_sel = np.append(best_roi_sel, x, axis=1)
            best_roi_sel_test = np.append(best_roi_sel_test, x_test, axis=1)


    #######################################
    # Save brain and image feature data
    #######################################
    # Analysis loop --------------------------------------------------------------
    print('----------------------------------------')
    print('Analysis loop')
    first = 1
    for feat, sbj in product(features_list, subjects_list):
        print('--------------------')
        print('Feature:    %s' % feat)
        print('Subject:    %s' % sbj)

        results_dir_alllayer_pca = os.path.join(results_dir_root, sbj)
        results_dir_alllayer_pca_test = os.path.join(results_dir_root_test, sbj)
        results_dir_alllayer_pca_aug = os.path.join(results_dir_root_aug, sbj)

        results_fmri_dir = os.path.join(results_fmri_root, sbj)
        # Preparing data
        print('Preparing data')

        # Brain data
        x = best_roi_sel[0:1200]      # Brain data
        x_labels = data_brain[sbj].select('image_index').flatten()  # Label (image index)
        x_labels = x_labels[0:1200]    # Label (image index)
        x_class = data_brain[sbj].select('Label') # Label (class index)
        WordNetID = data_brain[sbj].select('stimulus_id')  # Label (class index)
        WordNetID = WordNetID[0:1200,0]
        class_idx = x_class[0:1200, 1]

        x_test = best_roi_sel_test[1200:2950]          # Brain data
        x_labels_test = data_brain_test[sbj].select('image_index').flatten()  # Label (image index)
        x_labels_test = x_labels_test[1200:2950]   # Label (image index)
        x_class_test = data_brain_test[sbj].select('Label') # Label (class index)
        WordNetID_test = data_brain_test[sbj].select('stimulus_id')  # Label (class index)
        WordNetID_test = WordNetID_test[1200:2950,0]
        class_idx_test = x_class_test[1200:2950, 1]

        # Averaging test brain data
        x_labels_test_unique, indices = np.unique(x_labels_test, return_index=True)
        x_test_unique = np.vstack([np.mean(x_test[(np.array(x_labels_test) == lb).flatten(), :], axis=0) for lb in x_labels_test_unique])
        WordNetID_test_unique = WordNetID_test[indices]
        class_idx_test_unique = class_idx_test[indices]

        # Target features and image labels (file names)
        y = data_features.get_features(feat)  # Target DNN features
        y_labels = data_features.index        # Label (image index)
        y = np.reshape(y,(y.shape[0],-1))

        y_test = data_features_test.get_features(feat)  # Target DNN features
        y_labels_test = data_features_test.index        # Label (image index)
        y_test = np.reshape(y_test,(y_test.shape[0],-1))

        y_aug = data_features_aug.get_features(feat)  # Target DNN features
        y_labels_aug_temp = data_features_aug.labels        # Label (image index)
        y_labels_aug = []
        for it in y_labels_aug_temp:
            y_labels_aug.append(int(it.split('_')[0][1:]))
        y_labels_aug = np.array(y_labels_aug)
        y_aug = np.reshape(y_aug,(y_aug.shape[0],-1))

        # Calculate normalization parameters
        # Normalize X (fMRI data)
        x_mean = np.mean(x, axis=0)[np.newaxis, :]  # np.newaxis was added to match Matlab outputs
        x_norm = np.std(x, axis=0, ddof=1)[np.newaxis, :]

        # Normalize Y (DNN features)
        y_mean = np.mean(y, axis=0)[np.newaxis, :]
        y_norm = np.std(y, axis=0, ddof=1)[np.newaxis, :]

        # Y index to sort Y by X (matching samples)
        y_index = np.array([np.where(np.array(y_labels) == xl) for xl in x_labels]).flatten()
        y_index_test = np.array([np.where(np.array(y_labels_test) == xl) for xl in x_labels_test]).flatten()
        y_index_test_unique = np.array([np.where(np.array(y_labels_test) == xl) for xl in x_labels_test_unique]).flatten()

        # X preprocessing
        print('Normalizing X')
        x = (x - x_mean) / (x_norm+TINY)
        x[np.isinf(x)] = 0

        x_test = (x_test - x_mean) / (x_norm+TINY)
        x_test[np.isinf(x_test)] = 0
        x_test_unique = (x_test_unique - x_mean) / (x_norm+TINY)
        x_test_unique[np.isinf(x_test_unique)] = 0

        print('Doing PCA')
        ipca = PCA(n_components=0.99, random_state=seed)
        ipca.fit(x)
        x = ipca.transform(x)
        x_test = ipca.transform(x_test)
        x_test_unique = ipca.transform(x_test_unique)
        print(x.shape)

        # Y preprocessing
        print('Normalizing Y')
        y = (y - y_mean) / (y_norm+TINY)
        y[np.isinf(y)] = 0
        y_test = (y_test - y_mean) / (y_norm+TINY)
        y_test[np.isinf(y_test)] = 0
        y_aug = (y_aug - y_mean) / (y_norm+TINY)
        y_aug[np.isinf(y_aug)] = 0

        print('Doing PCA')
        ipca = PCA(n_components=0.99, random_state=seed)
        ipca.fit(y)
        # ipca.fit(y_aug)
        y = ipca.transform(y)
        y_test = ipca.transform(y_test)
        y_aug = ipca.transform(y_aug)
        print(y.shape)

        print('Sorting Y')
        y = y[y_index, :]
        y_test = y_test[y_index_test, :]
        y_test_unique = y_test[y_index_test_unique, :]

        if first:
            feat_pca_train = y
            feat_pca_test = y_test
            feat_pca_aug = y_aug
            feat_pca_test_unique = y_test_unique
            first = 0
        else:
            feat_pca_train = np.concatenate((feat_pca_train, y), axis=1)
            feat_pca_test = np.concatenate((feat_pca_test, y_test), axis=1)
            feat_pca_aug = np.concatenate((feat_pca_aug, y_aug), axis=1)
            feat_pca_test_unique = np.concatenate((feat_pca_test_unique, y_test_unique), axis=1)
        print(feat_pca_test_unique.shape)


        makedir_ifnot(results_dir_alllayer_pca)
        makedir_ifnot(results_dir_alllayer_pca_test)
        makedir_ifnot(results_dir_alllayer_pca_aug)
        results_dir_alllayer_pca_path = os.path.join(results_dir_alllayer_pca, "feat_pca_train.mat")
        io.savemat(results_dir_alllayer_pca_path, {"data":feat_pca_train})
        results_dir_alllayer_pca_test_path = os.path.join(results_dir_alllayer_pca_test, "feat_pca_test.mat")
        io.savemat(results_dir_alllayer_pca_test_path, {"data":feat_pca_test})
        results_dir_alllayer_pca_aug_path = os.path.join(results_dir_alllayer_pca_aug, "feat_pca_aug.mat")
        io.savemat(results_dir_alllayer_pca_aug_path, {"data":feat_pca_aug})
        results_dir_alllayer_pca_test_path = os.path.join(results_dir_alllayer_pca_test, "feat_pca_test_unique.mat")
        io.savemat(results_dir_alllayer_pca_test_path, {"data":feat_pca_test_unique})


        makedir_ifnot(results_fmri_dir)
        results_fmri_dir_path = os.path.join(results_fmri_dir, "fmri_train_data.mat")
        io.savemat(results_fmri_dir_path, {"data":x, "image_idx":x_labels, "WordNetID":WordNetID, "class_idx":class_idx})
        results_fmri_dir_path = os.path.join(results_fmri_dir, "fmri_test_data.mat")
        io.savemat(results_fmri_dir_path, {"data":x_test, "image_idx":x_labels_test, "WordNetID":WordNetID_test, "class_idx":class_idx_test})
        results_fmri_dir_path = os.path.join(results_fmri_dir, "fmri_test_data_unique.mat")
        io.savemat(results_fmri_dir_path, {"data":x_test_unique, "image_idx":x_labels_test_unique, "WordNetID":WordNetID_test_unique, "class_idx":class_idx_test_unique})

    #######################################
    # Save text feature data
    #######################################

    for feat, sbj in product(text_embedding_list, subjects_list):
        print('--------------------')
        print('Feature:    %s' % feat)
        print('Subject:    %s' % sbj)

        results_dir_text_fea = os.path.join(results_text_root, feat, sbj)
        results_dir_text_fea_test = os.path.join(results_text_root_test, feat, sbj)
        results_dir_text_fea_aug = os.path.join(results_text_root_aug, feat, sbj)
        # Preparing data
        print('Preparing data')

        # Brain data
        x_class = data_brain[sbj].select('Label')[0:1200]   # Label (class index)
        WordNetID = data_brain[sbj].select('stimulus_id')  # Label (class index)
        WordNetID = WordNetID[0:1200,0]
        class_idx = x_class[0:1200, 1]

        x_labels_test = data_brain_test[sbj].select('image_index').flatten()  # Label (image index)
        x_labels_test = x_labels_test[1200:2950]    # Label (image index)
        x_class_test = data_brain_test[sbj].select('Label')  # Label (class index)
        WordNetID_test = data_brain_test[sbj].select('stimulus_id')  # Label (class index)
        WordNetID_test = WordNetID_test[1200:2950,0]
        class_idx_test = x_class_test[1200:2950, 1]

        # Averaging test brain data
        x_labels_test_unique, indices = np.unique(x_labels_test, return_index=True)
        WordNetID_test_unique = WordNetID_test[indices]
        class_idx_test_unique = class_idx_test[indices]

        # Target text features and  wnid
        name = 'ImageNet_class200_' + feat + '.pkl'
        full = os.path.join(model_extracted_textual_features, name)
        dictionary = pickle.load(open(full, 'rb'))

        firstfeat = 1
        firstlabel = 1
        for key, value in dictionary.items():
            for k, v in value.items():
                # print(k, v)
                if k == 'wnid':
                    # print(v)
                    v = int(v[1:])
                    if firstlabel:
                        text_label = np.array([v])
                        firstlabel = 0
                    else:
                        text_label = np.concatenate((text_label, np.array([v])), axis=0)

                elif k == 'feats':
                    v = np.expand_dims(v, axis=0)
                    if firstfeat:
                        text_feat = v
                        firstfeat = 0
                    else:
                        text_feat = np.concatenate((text_feat, v), axis=0)

        # Target text features and  wnid
        name = 'ImageNet_trainval_classes_' + feat + '.pkl'
        full = os.path.join(model_extracted_textual_features, name)
        dictionary = pickle.load(open(full, 'rb'))

        firstfeat = 1
        firstlabel = 1
        for key, value in dictionary.items():
            for k, v in value.items():
                # print(k, v)
                if k == 'wnid':
                    # print(v)
                    v = int(v[1:])
                    if firstlabel:
                        text_label_aug = np.array([v])
                        firstlabel = 0
                    else:
                        text_label_aug = np.concatenate((text_label_aug, np.array([v])), axis=0)

                elif k == 'feats':
                    v = np.expand_dims(v, axis=0)
                    if firstfeat:
                        text_feat_aug = v
                        firstfeat = 0
                    else:
                        text_feat_aug = np.concatenate((text_feat_aug, v), axis=0)


        # t index to sort t by X (matching samples)
        t_index = np.array([np.where(np.array(text_label) == xl) for xl in WordNetID.astype(int)]).flatten()
        t_index_test = np.array([np.where(np.array(text_label) == xl) for xl in WordNetID_test.astype(int)]).flatten()
        t_index_test_unique = np.array([np.where(np.array(text_label) == xl) for xl in WordNetID_test_unique.astype(int)]).flatten()
        t_index_aug = np.array([np.where(np.array(text_label_aug) == xl) for xl in y_labels_aug]).flatten()

        print('Sorting text')
        t = text_feat[t_index, :]
        t_test = text_feat[t_index_test, :]
        t_aug = text_feat_aug[t_index_aug, :]
        t_test_unique = text_feat[t_index_test_unique, :]

        print(t.shape)
        print(t_test.shape)
        print(t_aug.shape)
        print(t_test_unique.shape)

        makedir_ifnot(results_dir_text_fea)
        makedir_ifnot(results_dir_text_fea_test)
        makedir_ifnot(results_dir_text_fea_aug)

        results_dir_text_fea_path = os.path.join(results_dir_text_fea, "text_feat_train.mat")
        io.savemat(results_dir_text_fea_path, {"data": t})

        results_dir_text_fea_test_path = os.path.join(results_dir_text_fea_test, "text_feat_test.mat")
        io.savemat(results_dir_text_fea_test_path, {"data": t_test})

        results_dir_text_fea_aug_path = os.path.join(results_dir_text_fea_aug, "text_feat_aug.mat")
        io.savemat(results_dir_text_fea_aug_path, {"data": t_aug})

        results_dir_text_fea_test_path = os.path.join(results_dir_text_fea_test, "text_feat_test_unique.mat")
        io.savemat(results_dir_text_fea_test_path, {"data": t_test_unique})

    print('%s finished.' % analysis_basename)