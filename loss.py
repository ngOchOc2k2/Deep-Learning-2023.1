import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=1.0):
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature

    def forward(self, positive_similarities, negative_similarities):
        # Tính softmax loss cho từng điểm dữ liệu trong batch
        batch_size = positive_similarities.size(0)

        # Tạo tensor chứa giá trị loss cho mỗi điểm dữ liệu trong batch
        losses = []

        for i in range(batch_size):
            # Tính softmax loss
            logits = torch.cat([positive_similarities[i].unsqueeze(0), negative_similarities[i]], dim=0)
            softmax_logits = F.log_softmax(logits / self.temperature, dim=0)

            # Loss chỉ tính cho positive
            loss_i = -softmax_logits[0].mean()
            losses.append(loss_i)

        # Chuyển danh sách losses thành tensor
        loss_tensor = torch.stack(losses)

        # Tính trung bình loss trên toàn bộ batch
        loss = loss_tensor.mean()

        return loss
    
    


class CrossEntropyLoss(nn.Module):
    def __init__(self):
        super(CrossEntropyLoss, self).__init__()

    def forward(self, softmax_positive, label_positive, softmax_negative, label_negative):
        # Chuyển các đầu vào thành tensor và chuyển đổi thành kiểu Long
        label_positive = torch.tensor(label_positive, dtype=torch.long)
        label_negative = torch.tensor(label_negative, dtype=torch.long)

        # Tính toán CrossEntropy loss cho positive
        positive_loss = nn.CrossEntropyLoss()(softmax_positive, label_positive)

        # Tính toán CrossEntropy loss cho negative
        negative_loss = nn.CrossEntropyLoss()(softmax_negative.view(-1, softmax_negative.size(-1)), label_negative.view(-1))

        # Tính trung bình loss
        mean_loss = (positive_loss + negative_loss) / 2

        return mean_loss


class CombineLoss(nn.Module):
    def __init__(self, temperature=1.0):
        super(CombineLoss, self).__init__()
        self.temperature = temperature
        self.ce_loss = nn.CrossEntropyLoss()
        
    def forward(self, softmax_positive, label_positive, softmax_negative, label_negative, positive_similarities, negative_similarities, device=None):

        # print(softmax_positive)
        # print(label_positive)
        # Tính toán CrossEntropy loss cho positive
        positive_loss = self.ce_loss(softmax_positive, label_positive)

        # Tính toán CrossEntropy loss cho negative
        negative_loss = self.ce_loss(softmax_negative.view(-1, softmax_negative.size(-1)), label_negative.view(-1))

        # Tính trung bình loss
        mean_loss = (positive_loss + negative_loss) / 2


        batch_size = positive_similarities.size(0)

        losses = []

        for i in range(batch_size):
            # Tính softmax loss
            logits = torch.cat([positive_similarities[i].unsqueeze(0), negative_similarities[i]], dim=0)
            softmax_logits = F.log_softmax(logits / self.temperature, dim=0)

            # Loss chỉ tính cho positive
            loss_i = -softmax_logits[0].mean()
            losses.append(loss_i)

        # Chuyển danh sách losses thành tensor
        loss_tensor = torch.stack(losses)

        # Tính trung bình loss trên toàn bộ batch
        loss = loss_tensor.mean()
        
        return mean_loss + loss