import torch

from tqdm.auto import tqdm
from torch.utils.tensorboard import SummaryWriter




def train_step(model: torch.nn.Module, 
               dataloader: torch.utils.data.DataLoader, 
               loss_fn: torch.nn.Module, 
               optimizer: torch.optim.Optimizer,
               device:torch.device):
    
    # Put model in train mode
    model.train()
    model.to(device)

    
    # Setup train loss and train accuracy values
    train_loss, train_acc = 0, 0

    # Loop through data loader data batches
    for i, (inputs, labels) in enumerate(dataloader, 0):

        inputs, labels = inputs.to(device), labels.to(device)

        #1.Forward pass
        # print("[][][][][][][]:",model)
        outputs = model(inputs)

        # print("outputs.shape:",outputs.shape)

        #2.Calculate and accumulate the loss 
        loss = loss_fn(outputs, labels.float())
        train_loss += loss.item()

        #3.Optimizer zero grad
        optimizer.zero_grad()
        
        #4.Loss backward
        loss.backward()
        
        #5.Optimizer step
        optimizer.step()

        train_loss += loss.item()

        # Calculate and accumulate accuracy metric across all batches
        y_pred_class = torch.argmax(torch.softmax(outputs, dim=1), dim=1)
        
        labels = labels.to(torch.float32)  # Convert labels to float32
        arglabels = torch.argmax(labels,dim=1)
        train_acc += (y_pred_class == arglabels).sum().item()/ len(outputs)
    
    # Adjust metrics to get average loss and accuracy per batch 
    train_loss = train_loss / len(dataloader)
    train_acc = train_acc / len(dataloader)

    return train_loss, train_acc


def test_step(model: torch.nn.Module, 
              dataloader: torch.utils.data.DataLoader, 
              loss_fn: torch.nn.Module,
              device:torch.device):
    #put model in eval mode
    model.eval()
    model.to(device)
    # Setup test loss and test accuracy values
    test_loss, test_acc = 0, 0

    # Turn on inference context manager
    with torch.inference_mode():
        # Loop through DataLoader batches
        for data in dataloader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            #1.forward pass
            outputs = model(inputs)

            #2.Calculate and accumulate loss


            loss = loss_fn(outputs, labels.float())
            test_loss += loss.item()

            # Calculate and accumulate accuracy
            _, predicted = torch.max(outputs.data, 1)
            _,labels_1_dim = torch.max(labels.data, 1)

            # print("label 0 is:",labels.size(0))#   = batch_size
            test_acc += ((predicted == labels_1_dim).sum().item()/len(predicted))
    # Adjust metrics to get average loss and accuracy per batch 
    test_loss = test_loss / len(dataloader)
    test_acc = test_acc / len(dataloader)
    return test_loss, test_acc
 


def train(model: torch.nn.Module, 
          train_dataloader: torch.utils.data.DataLoader, 
          test_dataloader: torch.utils.data.DataLoader, 
          optimizer: torch.optim.Optimizer,
          loss_fn: torch.nn.Module,
          epochs: int,
          device:torch.device,
          writer: torch.utils.tensorboard.writer.SummaryWriter):

    results = {"train_loss": [],
    "train_acc": [],
    "test_loss": [],
    "test_acc": []
    }

    for epoch in tqdm(range(epochs)):
        train_loss, train_acc = train_step(model=model,
                                           dataloader=train_dataloader,
                                           loss_fn=loss_fn,
                                           optimizer=optimizer,
                                           device=device)
        test_loss, test_acc = test_step(model=model,
                                        dataloader=test_dataloader,
                                        loss_fn=loss_fn,
                                        device=device)
        print(
            f"Epoch: {epoch+1} | "
            f"train_loss: {train_loss:.4f} | "
            f"train_acc: {train_acc:.4f} | "
            f"test_loss: {test_loss:.4f} | "
            f"test_acc: {test_acc:.4f}"
        )

        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)


            ### New: Use the writer parameter to track experiments ###
        # See if there's a writer, if so, log to it
        if writer:
            # Add results to SummaryWriter
            writer.add_scalars(main_tag="Loss", 
                               tag_scalar_dict={"train_loss": train_loss,
                                                "test_loss": test_loss},
                               global_step=epoch)
            writer.add_scalars(main_tag="Accuracy", 
                               tag_scalar_dict={"train_acc": train_acc,
                                                "test_acc": test_acc}, 
                               global_step=epoch)

            # Close the writer
            writer.close()
        else:
            pass   
    
    ### End new ###
    return results


