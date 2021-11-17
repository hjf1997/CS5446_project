import torch
import torch.nn as nn
import torch.nn.functional as F
from config import gamma



class QNet(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(QNet, self).__init__()
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs

        self.fc1 = nn.Linear(num_inputs, 128)
        self.fc2 = nn.Linear(128, num_outputs)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform(m.weight)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        qvalue = self.fc2(x)
        return qvalue

    @classmethod
    def train_model(cls, online_net, target_net, optimizer, batch, device):
        states = torch.cat(batch.state, dim=0)
        next_states = torch.cat(batch.next_state, dim=0)
        actions = torch.Tensor(batch.action).to(device).float()
        rewards = torch.Tensor(batch.reward).to(device)
        masks = torch.Tensor(batch.mask).to(device)

        pred = online_net(states)
        next_pred = target_net(next_states)

        pred = torch.sum(pred.mul(actions), dim=1)

        target = rewards + masks * gamma * next_pred.max(1)[0]

        loss = F.mse_loss(pred, target.detach())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return loss

    def get_action(self, input):
        qvalue = self.forward(input)
        _, action = torch.max(qvalue, 1)
        return action.cpu().numpy()[0]

class DoubleDQNet(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(DoubleDQNet, self).__init__()
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs

        self.fc1 = nn.Linear(num_inputs, 128)
        self.fc2 = nn.Linear(128, num_outputs)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform(m.weight)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        qvalue = self.fc2(x)
        return qvalue

    @classmethod
    def train_model(cls, online_net, target_net, optimizer, batch, device):
        states = torch.cat(batch.state, dim=0)
        next_states = torch.cat(batch.next_state, dim=0)
        actions = torch.Tensor(batch.action).to(device).float()
        rewards = torch.Tensor(batch.reward).to(device)
        masks = torch.Tensor(batch.mask).to(device)

        pred = online_net(states)
        _, action_from_online_net = online_net(next_states).max(1)
        next_pred = target_net(next_states)

        pred = torch.sum(pred.mul(actions), dim=1)

        target = rewards + masks * gamma * next_pred.gather(1, action_from_online_net.unsqueeze(1)).squeeze(1)

        loss = F.mse_loss(pred, target.detach())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return loss

    def get_action(self, input):
        qvalue = self.forward(input)
        _, action = torch.max(qvalue, 1)
        return action.cpu().numpy()[0]


class DuelDQNet(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(DuelDQNet, self).__init__()
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs

        self.fc = nn.Linear(num_inputs, 128)
        self.fc_adv = nn.Linear(128, num_outputs)
        self.fc_val = nn.Linear(128, 1)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform(m.weight)

    def forward(self, x):
        x = F.relu(self.fc(x))
        adv = self.fc_adv(x)
        adv = adv.view(-1, self.num_outputs)
        val = self.fc_val(x)
        val = val.view(-1, 1)

        qvalue = val + (adv - adv.mean(dim=1, keepdim=True))
        return qvalue

    @classmethod
    def train_model(cls, online_net, target_net, optimizer, batch, device):
        states = torch.stack(batch.state)
        next_states = torch.stack(batch.next_state)
        actions = torch.Tensor(batch.action).to(device).float()
        rewards = torch.Tensor(batch.reward).to(device)
        masks = torch.Tensor(batch.mask).to(device)

        pred = online_net(states).squeeze(1)
        next_pred = target_net(next_states).squeeze(1)

        pred = torch.sum(pred.mul(actions), dim=1)

        target = rewards + masks * gamma * next_pred.max(1)[0]


        loss = F.mse_loss(pred, target.detach())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return loss

    def get_action(self, input):
        qvalue = self.forward(input)
        _, action = torch.max(qvalue, 1)
        return action.cpu().numpy()[0]


class PerNet(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(PerNet, self).__init__()
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs

        self.fc1 = nn.Linear(num_inputs, 128)
        self.fc2 = nn.Linear(128, num_outputs)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform(m.weight)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        qvalue = self.fc2(x)
        return qvalue

    @classmethod
    def get_td_error(cls, online_net, target_net, state, next_state, action, reward, mask, device):
        state = torch.stack(state)
        next_state = torch.stack(next_state)
        action = torch.Tensor(action).to(device)
        reward = torch.Tensor(reward).to(device)
        mask = torch.Tensor(mask).to(device)

        pred = online_net(state).squeeze(1)
        next_pred = target_net(next_state).squeeze(1)

        pred = torch.sum(pred.mul(action), dim=1)

        target = reward + mask * gamma * next_pred.max(1)[0]

        td_error = pred - target.detach()

        return td_error

    @classmethod
    def train_model(cls, online_net, target_net, optimizer, batch, weights, device):
        td_error = cls.get_td_error(online_net, target_net, batch.state, batch.next_state, batch.action, batch.reward, batch.mask, device)
        loss = pow(td_error, 2) * weights
        loss = loss.mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return loss

    def get_action(self, input):
        qvalue = self.forward(input)
        _, action = torch.max(qvalue, 1)
        return action.cpu().numpy()[0]