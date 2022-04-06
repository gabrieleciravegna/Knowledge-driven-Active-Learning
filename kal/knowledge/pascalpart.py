from typing import Union, Tuple

import numpy as np
import torch

from . import KnowledgeLoss


class PascalPartLoss(KnowledgeLoss):
    def __init__(self, names=None, scale="none", mu=10):
        super().__init__(names)
        self.scale = scale
        self.mu = mu

    def __call__(self, output, targets=False,
                 return_arg_max=False) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:

        Aeroplane = output[:, 0]
        Aeroplane_Body = output[:, 1]
        Arm = output[:, 2]
        Backside = output[:, 3]
        Beak = output[:, 4]
        Bicycle = output[:, 5]
        Bird = output[:, 6]
        Boat = output[:, 7]
        Bottle = output[:, 8]
        Bottle_Body = output[:, 9]
        Bus = output[:, 10]
        Cap = output[:, 11]
        Car = output[:, 12]
        Cat = output[:, 13]
        Chainwheel = output[:, 14]
        Chair = output[:, 15]
        Coach = output[:, 16]
        Cow = output[:, 17]
        Dog = output[:, 18]
        Door = output[:, 19]
        Ear = output[:, 20]
        Ebrow = output[:, 21]
        Engine = output[:, 22]
        Eye = output[:, 23]
        Foot = output[:, 24]
        Frontside = output[:, 25]
        Hair = output[:, 26]
        Hand = output[:, 27]
        Handlebar = output[:, 28]
        Head = output[:, 29]
        Headlight = output[:, 30]
        Hoof = output[:, 31]
        Horn = output[:, 32]
        Horse = output[:, 33]
        Leftside = output[:, 34]
        Leg = output[:, 35]
        Mirror = output[:, 36]
        Motorbike = output[:, 37]
        Mouth = output[:, 38]
        Muzzle = output[:, 39]
        Neck = output[:, 40]
        Nose = output[:, 41]
        Paw = output[:, 42]
        Person = output[:, 43]
        Plant = output[:, 44]
        Plate = output[:, 45]
        Pot = output[:, 46]
        Pottedplant = output[:, 47]
        Rightside = output[:, 48]
        Roofside = output[:, 49]
        Saddle = output[:, 50]
        Screen = output[:, 51]
        Sheep = output[:, 52]
        Sofa = output[:, 53]
        Stern = output[:, 54]
        Table = output[:, 55]
        Tail = output[:, 56]
        Torso = output[:, 57]
        Train = output[:, 58]
        Train_Head = output[:, 59]
        Tvmonitor = output[:, 60]
        Wheel = output[:, 61]
        Window = output[:, 62]
        Wing = output[:, 63]

        loss_fol_product_tnorm = []

        # if a:
        loss_fol_product_tnorm.extend([
            # A: OBJECT-PART --> [OBJECTS] RULES
            # 0) 'Screen': ['Tvmonitor'],
            (Screen * (1 - Tvmonitor)),
            # 1) 'Coach': ['Train'],
            (Coach * (1 - Train)),
            # 2) 'Torso': ['Person', 'Horse', 'Cow', 'Dog', 'Bird', 'Cat', 'Sheep'],
            (Torso * (1 - Person) * (1 - Horse) * (1 - Cow) * (1 - Dog) * (1 - Bird) * (1 - Cat) * (1 - Sheep)),
            # 3) 'Leg': ['Person', 'Horse', 'Cow', 'Dog', 'Bird', 'Cat', 'Sheep'],
            (Leg * (1 - Person) * (1 - Horse) * (1 - Cow) * (1 - Dog) * (1 - Bird) * (1 - Cat) * (1 - Sheep)),
            # 4) 'Head': ['Person', 'Horse', 'Cow', 'Dog', 'Bird', 'Cat', 'Sheep'],
            (Head * (1 - Person) * (1 - Horse) * (1 - Cow) * (1 - Dog) * (1 - Bird) * (1 - Cat) * (1 - Sheep)),
            # 5) 'Ear': ['Person', 'Horse', 'Cow', 'Dog', 'Cat', 'Sheep'],
            (Ear * (1 - Person) * (1 - Horse) * (1 - Cow) * (1 - Dog) * (1 - Cat) * (1 - Sheep)),
            # 6) 'Eye': ['Person', 'Cow', 'Dog', 'Bird', 'Cat', 'Horse', 'Sheep'],
            (Eye * (1 - Person) * (1 - Cow) * (1 - Dog) * (1 - Bird) * (1 - Cat) * (1 - Horse) * (1 - Sheep)),
            # 7) 'Ebrow': ['Person'],
            (Ebrow * (1 - Person)),
            # 8) 'Mouth': ['Person'],
            (Mouth * (1 - Person)),
            # 9) 'Hair': ['Person'],
            (Hair * (1 - Person)),
            # 10) 'Nose': ['Person', 'Dog', 'Cat'],
            (Nose * (1 - Person) * (1 - Dog) * (1 - Cat)),
            # 11) 'Neck': ['Person', 'Horse', 'Cow', 'Dog', 'Bird', 'Cat', 'Sheep'],
            (Neck * (1 - Person) * (1 - Horse) * (1 - Cow) * (1 - Dog) * (1 - Bird) * (1 - Cat) * (1 - Sheep)),
            # 12) 'Arm': ['Person'],
            (Arm * (1 - Person)),
            # 13) 'Muzzle': ['Horse', 'Cow', 'Dog', 'Sheep'],
            (Muzzle * (1 - Horse) * (1 - Cow) * (1 - Dog) * (1 - Sheep)),
            # 14) 'Hoof': ['Horse'],
            (Hoof * (1 - Horse)),
            # 15) 'Tail': ['Horse', 'Cow', 'Dog', 'Bird', 'Sheep', 'Cat', 'Aeroplane'],
            (Tail * (1 - Horse) * (1 - Cow) * (1 - Dog) * (1 - Bird) * (1 - Sheep) * (1 - Cat) * (1 - Aeroplane)),
            # 16) 'Bottle_Body': ['Bottle'],
            (Bottle_Body * (1 - Bottle)),
            # 17) 'Paw': ['Dog', 'Cat'],
            (Paw * (1 - Dog) * (1 - Cat)),
            # 18) 'Aeroplane_Body': ['Aeroplane'],
            (Aeroplane_Body * (1 - Aeroplane)),
            # 19) 'Wing': ['Aeroplane', 'Bird'],
            (Wing * (1 - Aeroplane) * (1 - Bird)),
            # 20) 'Wheel': ['Aeroplane', 'Car', 'Bicycle', 'Bus', 'Motorbike'],
            (Wheel * (1 - Aeroplane) * (1 - Car) * (1 - Bicycle) * (1 - Bus) * (1 - Motorbike)),
            # 21) 'Stern': ['Aeroplane'],
            (Stern * (1 - Aeroplane)),
            # 22) 'Cap': ['Bottle'],
            (Cap * (1 - Bottle)),
            # 23) 'Hand': ['Person'],
            (Hand * (1 - Person)),
            # 24) 'Frontside': ['Car', 'Bus', 'Train'],
            (Frontside * (1 - Car) * (1 - Bus) * (1 - Train)),
            # 25) 'Rightside': ['Car', 'Bus', 'Train'],
            (Rightside * (1 - Car) * (1 - Bus) * (1 - Train)),
            # 26) 'Roofside': ['Car', 'Bus', 'Train'],
            (Roofside * (1 - Car) * (1 - Bus) * (1 - Train)),
            # 27) 'Backside': ['Car', 'Bus', 'Train'],
            (Backside * (1 - Car) * (1 - Bus) * (1 - Train)),
            # 28) 'Leftside': ['Car', 'Train', 'Bus'],
            (Leftside * (1 - Car) * (1 - Bus) * (1 - Train)),
            # 29) 'Door': ['Car', 'Bus'],
            (Door * (1 - Car) * (1 - Bus)),
            # 30) 'Mirror': ['Car', 'Bus'],
            (Mirror * (1 - Car) * (1 - Bus)),
            # 31) 'Headlight': ['Car', 'Bus', 'Train', 'Motorbike', 'Bicycle'],
            (Headlight * (1 - Car) * (1 - Bus) * (1 - Train) * (1 - Motorbike) * (1 - Bicycle)),
            # 32) 'Window': ['Car', 'Bus'],
            (Window * (1 - Car) * (1 - Bus)),
            # 33) 'Plate': ['Car', 'Bus'],
            (Plate * (1 - Car) * (1 - Bus)),
            # 34) 'Engine': ['Aeroplane'],
            (Engine * (1 - Aeroplane)),
            # 35) 'Foot': ['Person', 'Bird'],
            (Foot * (1 - Person) * (1 - Bird)),
            # 36) 'Chainwheel': ['Bicycle'],
            (Chainwheel * (1 - Bicycle)),
            # 37) 'Saddle': ['Bicycle', 'Motorbike'],
            (Saddle * (1 - Bicycle) * (1 - Motorbike)),
            # 38) 'Handlebar': ['Bicycle', 'Motorbike'],
            (Handlebar * (1 - Bicycle) * (1 - Motorbike)),
            # 39) 'Train_Head': ['Train'],
            (Train_Head * (1 - Train)),
            # 40) 'Beak': ['Bird'],
            (Beak * (1 - Bird)),
            # 41) 'Pot': ['Pottedplant'],
            (Pot * (1 - Pottedplant)),
            # 42) 'Plant': ['Pottedplant'],
            (Plant * (1 - Pottedplant)),
            # 43) 'Horn': ['Cow', 'Sheep']
            (Horn * (1 - Cow) * (1 - Sheep))]
        )

        # if b:
        loss_fol_product_tnorm.extend([
            # B: OBJECT --> [OBJECT-PARTS] RULES
            # 44) Tvmonitor = ['Screen'],
            (Tvmonitor * (1 - Screen)),
            # 45) Train = ['Coach', 'Leftside', 'Train_Head', 'Headlight', 'Frontside', 'Rightside', 'Backside',
            # 'Roofside'],
            (Train * (1 - Coach) * (1 - Leftside) * (1 - Train_Head) * (1 - Headlight) * (1 - Frontside) * (
                        1 - Rightside) * (1 - Backside) * (1 - Roofside)),
            # 46) Person = ['Torso', 'Leg', 'Head', 'Ear', 'Eye', 'Ebrow', 'Mouth', 'Hair', 'Nose', 'Neck', 'Arm',
            # 'Hand', 'Foot'],
            (Person * (1 - Torso) * (1 - Leg) * (1 - Head) * (1 - Ear) * (1 - Eye) * (1 - Ebrow) * (1 - Mouth) * (
                        1 - Hair) * (1 - Nose) * (1 - Neck) * (1 - Arm) * (1 - Hand) * (1 - Foot)),
            # Boat = [],
            # 47) Horse = ['Head', 'Ear', 'Muzzle', 'Torso', 'Neck', 'Leg', 'Hoof', 'Tail', 'Eye'],
            (Horse * (1 - Head) * (1 - Ear) * (1 - Muzzle) * (1 - Torso) * (1 - Neck) * (1 - Leg) * (1 - Hoof) * (
                        1 - Tail) * (1 - Eye)),
            # 48) Cow = ['Head', 'Ear', 'Eye', 'Muzzle', 'Torso', 'Neck', 'Leg', 'Tail', 'Horn'],
            (Cow * (1 - Head) * (1 - Ear) * (1 - Eye) * (1 - Muzzle) * (1 - Torso) * (1 - Neck) * (1 - Leg) * (
                        1 - Tail) * (1 - Horn)),
            # 49) Bottle = ['Bottle_Body', 'Cap'],
            (Bottle * (1 - Bottle_Body) * (1 - Cap)),
            # 50) Dog = ['Head', 'Ear', 'Torso', 'Neck', 'Leg', 'Paw', 'Eye', 'Muzzle', 'Nose', 'Tail'],
            (Dog * (1 - Head) * (1 - Ear) * (1 - Torso) * (1 - Neck) * (1 - Leg) * (1 - Paw) * (1 - Eye) * (
                        1 - Muzzle) * (1 - Nose) * (1 - Tail)),
            # 51) Aeroplane = ['Aeroplane_Body', 'Wing', 'Wheel', 'Stern', 'Engine', 'Tail'],
            (Aeroplane * (1 - Aeroplane_Body) * (1 - Wing) * (1 - Wheel) * (1 - Stern) * (1 - Engine) * (1 - Tail)),
            # 52) Car = ['Frontside', 'Rightside', 'Door', 'Mirror', 'Headlight', 'Wheel', 'Window', 'Plate',
            # 'Roofside', 'Backside', 'Leftside'],
            (Car * (1 - Frontside) * (1 - Rightside) * (1 - Door) * (1 - Mirror) * (1 - Headlight) * (1 - Wheel) * (
                        1 - Window) * (1 - Plate) * (1 - Roofside) * (1 - Backside) * (1 - Leftside)),
            # 53) Bus = ['Plate', 'Frontside', 'Rightside', 'Door', 'Mirror', 'Headlight', 'Window', 'Wheel',
            # 'Leftside', 'Backside', 'Roofside'],
            (Bus * (1 - Plate) * (1 - Frontside) * (1 - Rightside) * (1 - Door) * (1 - Mirror) * (1 - Headlight) * (
                        1 - Window) * (1 - Wheel) * (1 - Leftside) * (1 - Backside) * (1 - Roofside)),
            # 54) Bicycle = ['Wheel', 'Chainwheel', 'Saddle', 'Handlebar', 'Headlight'],
            (Bicycle * (1 - Wheel) * (1 - Chainwheel) * (1 - Saddle) * (1 - Handlebar) * (1 - Headlight)),
            # Table = [],
            # Chair = [],
            # 55) Bird = ['Head', 'Eye', 'Beak', 'Torso', 'Neck', 'Leg', 'Foot', 'Tail', 'Wing'],
            (Bird * (1 - Head) * (1 - Eye) * (1 - Beak) * (1 - Torso) * (1 - Neck) * (1 - Leg) * (1 - Foot) * (
                        1 - Tail) * (1 - Wing)),
            # 56) Cat = ['Head', 'Ear', 'Eye', 'Nose', 'Torso', 'Neck', 'Leg', 'Paw', 'Tail'],
            (Cat * (1 - Head) * (1 - Ear) * (1 - Eye) * (1 - Nose) * (1 - Torso) * (1 - Neck) * (1 - Leg) * (
                        1 - Paw) * (1 - Tail)),
            # 57) Motorbike = ['Wheel', 'Headlight', 'Handlebar', 'Saddle'],
            (Motorbike * (1 - Wheel) * (1 - Headlight) * (1 - Handlebar) * (1 - Saddle)),
            # 58) Sheep = ['Head', 'Ear', 'Eye', 'Muzzle', 'Torso', 'Neck', 'Leg', 'Tail', 'Horn'],
            (Sheep * (1 - Head) * (1 - Ear) * (1 - Eye) * (1 - Muzzle) * (1 - Torso) * (1 - Neck) * (1 - Leg) * (
                        1 - Tail) * (1 - Horn)),
            # Sofa = [],
            # 59) Pottedplant = ['Pot', 'Plant']
            (Pottedplant * (1 - Pot) * (1 - Plant)),
        ])

        # if c:
        loss_fol_product_tnorm.extend([
            # 60) C: OR ON THE OBJECTS
            ((1 - Tvmonitor) * (1 - Train) * (1 - Person) * (1 - Boat) * (1 - Horse) * (1 - Cow) * (1 - Bottle) *
             (1 - Dog) * (1 - Aeroplane) * (1 - Car) * (1 - Bus) * (1 - Bicycle) * (1 - Table) * (1 - Chair) *
             (1 - Bird) * (1 - Cat) * (1 - Motorbike) * (1 - Sheep) * (1 - Sofa) * (1 - Pottedplant)),
        ])

        # D: OR ON THE PARTS ((1 - Screen) *  (1 - Coach) *  (1 - Torso) * (1 - Leg) * (1 - Head) * (1 - Ear) * (1 -
        # Eye) * (1 - Ebrow) * (1 - Mouth) * (1 - Hair) * (1 - Nose) * (1 - Neck) * (1 - Arm) * (1 - Muzzle) * (1 -
        # Hoof) * (1 - Tail) * (1 - Bottle_Body) * (1 - Paw) * (1 - Aeroplane_Body) * (1 - Wing) * (1 - Wheel) * (1 -
        # Stern) * (1 - Cap) * (1 - Hand) * (1 - Frontside) * (1 - Rightside) * (1 - Door) * (1 - Mirror) * (1 -
        # Headlight) * (1 - Window) * (1 - Plate) * (1 - Roofside) * (1 - Backside) * (1 - Leftside) * (1 - Engine) *
        # (1 - Foot) * (1 - Chainwheel) * (1 - Saddle) * (1 - Handlebar) * (1 - Train_Head) * (1 - Beak) * (1 - Pot)
        # * (1 - Plant) * (1 - Horn))

        losses = torch.stack(loss_fol_product_tnorm, dim=0)

        if self.scale:
            if self.scale == "a" or self.scale == "both":
                # scale the first rules for the number of predictions made (they may become noisy)
                num_preds = (output > 0.5).sum(dim=1)
                scaling = np.ones(output.shape[0]) / (num_preds + 1)  # to avoid numerical problem
                scaled_losses = losses[:44] * scaling
                losses[:44] = scaled_losses
            if self.scale == "c" or self.scale == "both":
                # scale by a factor 10 the last rule (which is the most important)
                losses[-1] = losses[-1] * self.mu

        losses = torch.sum(losses, dim=1)

        loss_sum = torch.squeeze(torch.sum(losses, dim=0))

        threshold = 0.5 if targets else 10.
        self.check_loss(output, losses, loss_sum, threshold)

        # print("Output", output)
        # print("Losses", losses)
        # print("Loss_sum", loss_sum)

        if return_arg_max:
            arg_max = torch.argmax(losses, dim=0)
            return loss_sum, arg_max
        return loss_sum
