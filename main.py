
from train import start_train
from evaluate import eval_model
import pkgutil
if __name__ == "__main__":

    ################## train individual model

    # model_name=['senet154', 'se_resnet152', 'se_resnext101_32x4d', 'resnet152', 'densenet201']

    # for model_n in model_name:
            
    #     dataloaders, dataset_sizes, class_names = get_data_loaders()

    #     model = build_model(model_n)
    #     if use_gpu:
    #         model = model.cuda()

    #     criterion = nn.CrossEntropyLoss()

    #     # Observe that only parameters of final layer are being optimized as
    #     # opoosed to before.
    #     optimizer_conv = optim.Adam(model.last_linear.parameters(), lr=0.01)

    #     # Decay LR by a factor of 0.1 every 7 epochs
    #     exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=7, gamma=0.01)

    #     model = train_model(model, criterion, optimizer_conv, exp_lr_scheduler, 50, dataloaders, dataset_sizes)
        # ... after training, save your model 
        # model.save_state_dict('%s.pt' %(model_n))
        # torch.save(model.state_dict(), '%s.pth' %(model_n))
        # .. to load your previously training model:
        # model.load_state_dict(torch.load('mytraining.pt'))

        # modulelist = list(model.last_linear.modules())
        # print(modulelist[-1].out_features)

    ################## train whole model

    
    # model = model_all_4096_512.build_whole_model()

    # model = start_train(modelif __name__ == "__main__":
#     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#     model = model_all_2048_512_5.build_whole_model()

#     PATH = os.path.abspath(os.path.dirname(__file__))

#     path_to_model = os.path.join(PATH, 'pretrained_model')

#     model.load_state_dict(torch.load(os.path.join(path_to_model, '%s.pth' %(model.name))))
#     model.to(device)

#     eval_model(model)if __name__ == "__main__":
#     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#     model = model_all_2048_512_5.build_whole_model()

#     PATH = os.path.abspath(os.path.dirname(__file__))

#     path_to_model = os.path.join(PATH, 'pretrained_model')

#     model.load_state_dict(torch.load(os.path.join(path_to_model, '%s.pth' %(model.name))))
#     model.to(device)

#     eval_model(model))
    
    # ... after training, save your model 
    # print([name for finder, name, _ in pkgutil.iter_modules(['model'])])

    # for finder, name, _ in pkgutil.iter_modules(['model']):
    #     print(name)
    #     mod = finder.find_module(name).load_module(name)
    #     model = mod.build_whole_model()
    #     model = start_train(model)
    #     eval_model(model)
    for finder, name, _ in pkgutil.iter_modules(['model']):
        if(name == 'model_128_7_35_32_7'):
            mod = finder.find_module(name).load_module(name)
            model = mod.build_whole_model()
            model = start_train(model)
            eval_model(model)

 


    ########
 