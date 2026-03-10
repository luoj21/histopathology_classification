import torch
import torch.optim as optim
import torch.nn as nn
import torch.optim.lr_scheduler as lr_scheduler
import json
import copy

from tqdm import tqdm
from src.nnet.torchBaselineModel import BaselineModel, EarlyStopping
from torch.utils.data import DataLoader
from src.nnet.data_gen import ImageDataset
from src.utils.my_utils import train_test_split, check_cuda_availability
from src.utils.logger import setup_logger
from src.utils.plots import plot_loss_acc, plot_roc_curve, plot_confusion_matrix


def main(test = False):
    """ Main function to train the model. If test is true, will run model on the test set"""
   
    with open("config.json", "r") as f:
        config = json.load(f)

    NUM_EPOCHS = config["model_params"]["epochs"]
    BATCH_SIZE = config["model_params"]["batch_size"]
    train_logger = setup_logger(config["paths"]["log_dir"])
    data_root = config["paths"]["data_root"]
    metadata_file = config["paths"]["metadata_file"]
    optimizer_name = config["optimizer"]

    train_losses, val_losses = [], []
    train_accs, val_accs= [], []
    epochs = []



    device = check_cuda_availability()
    model = BaselineModel(num_classes=config["model_params"]["num_classes"], 
                          num_channels=config["model_params"]["num_channels"]).to(device)

    dataset= ImageDataset(data_root, metadata_file, 
                          augment=config['preprocessing']['augment'], 
                          ycbcr=config['preprocessing']['ycbcr'], 
                          resize=config['preprocessing']['resize'])
                          
    train_dataset, val_dataset, test_dataset = train_test_split(dataset, 0.7, 0.15)

    train_no_augment = copy.deepcopy(train_dataset) # Original training data with no augmentation

    # 2 Different augmented train sets
    train_dataset_augment1 = copy.deepcopy(train_dataset)
    train_dataset_augment1.dataset.augment = True 
    train_dataset_augment2 = copy.deepcopy(train_dataset)
    train_dataset_augment2.dataset.augment = True

    
    full_train_dataset = torch.utils.data.ConcatDataset([train_no_augment, train_dataset_augment1, train_dataset_augment2])

    
    train_loader = DataLoader(full_train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=1, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=1, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=1, pin_memory=True)

    
    if optimizer_name == "Adam":
           optimizer = optim.Adam(model.parameters(), 
                                  lr=config["model_params"]["learning_rate"], 
                                  weight_decay = config["model_params"]["weight_decay"])
           
    elif optimizer_name == "SGD":
        optimizer = optim.SGD(model.parameters(), 
                              lr=config["model_params"]["learning_rate"], 
                              momentum=config["model_params"]["momentum"], 
                              weight_decay = config["model_params"]["weight_decay"])
    
    
    scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.25)
    criterion = nn.CrossEntropyLoss()
   
    es = EarlyStopping(patience=config["model_params"]["early_stopping_patience"], 
                       min_delta=0,
                        restore_best_weights=True)

    # Training Loop
    for epoch in tqdm(range(NUM_EPOCHS), desc = "### Training Loop ### ", leave = False):

        epochs.append(epoch)
        tqdm.write(f"Epoch {epoch + 1} / {NUM_EPOCHS}")
        model.train()
        running_loss = 0.0
        total_correct = 0

        for images, labels in tqdm(train_loader):

            optimizer.zero_grad() # clear gradients
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            _, preds_class = torch.max(outputs, 1)
            
            loss = criterion(outputs, labels) 
            loss.backward()

            optimizer.step()
            
            running_loss += loss.item() * images.size(0)
            total_correct += (preds_class == labels).sum().item()


        train_loss = running_loss / len(train_loader.dataset)
        train_losses.append(train_loss)

        train_acc = total_correct / len(train_loader.dataset)
        train_accs.append(train_acc)

        # Validation phase
        model.eval()
        running_loss = 0.0
        total_correct = 0

        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc =  "### Validation Loop ###"):

                images, labels = images.to(device), labels.to(device)

                outputs = model(images)
                _, preds_class = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

                running_loss += loss.item() * images.size(0)
                total_correct += (preds_class == labels).sum().item()


        val_loss = running_loss / len(val_loader.dataset)
        val_losses.append(val_loss)

        val_acc = total_correct / len(val_loader.dataset)
        val_accs.append(val_acc)

        train_logger.info(f"Epoch: {(epoch+1)} / {NUM_EPOCHS}. Train acc: {train_acc}. Val acc: {val_acc}. Train loss: {train_loss}. Val loss: {val_loss}. LR: {optimizer.param_groups[0]['lr']} \n")

        
        es(model, val_loss)
        if es.early_stop:
            train_logger.info(es.status)
            plot_loss_acc(epochs, train_losses, val_losses, train_accs, val_accs, output_dir = config["paths"]["plots_dir"])
            torch.save(es.best_model, f"{config['paths']['model_out_dir']}\\{config['model_params']['model_name']}")
            break

        scheduler.step()

    # Run model on test set
    if test:
        model.eval()
        running_loss = 0.0
        total_correct = 0
        y_true, y_preds, y_preds_proba = [], [], []

        with torch.no_grad():
            for images, labels in tqdm(test_loader, desc =  "### Testing Loop ###"):

                images, labels = images.to(device), labels.to(device)

                outputs = model(images)
                _, preds_class = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

                running_loss += loss.item() * images.size(0)
                total_correct += (preds_class == labels).sum().item()

                y_true.extend(labels.cpu().numpy())
                y_preds.extend(preds_class.cpu().numpy())
                y_preds_proba.extend(torch.nn.functional.softmax(outputs, dim=1).cpu().numpy())


        test_loss = running_loss / len(test_loader.dataset)
        test_acc = total_correct / len(test_loader.dataset)

        train_logger.info(f"Test acc: {test_acc}. Test loss: {test_loss} \n")


        plot_confusion_matrix(y_preds, y_true, class_names = dataset.img_types, output_dir = config["paths"]["plots_dir"])
        plot_roc_curve(y_preds_proba, y_true, class_names = dataset.img_types, output_dir = config["paths"]["plots_dir"])




if __name__ == "__main__":
    main(test = True)


    


