from ghd.losses.diceloss import MultiClassDiceLoss
from ghd.losses.meshloss import Mesh_loss
from ghd.losses.meshloss_oa import Mesh_loss_opening_alignment
from ghd.losses.meshloss_do import Mesh_loss_differentiable_occupancy
from ghd.losses.meshloss_dc import Mesh_loss_do_differentiable_centreline
from ghd.losses.meshloss_pouch import Mesh_loss_pouch_only

diceloss = MultiClassDiceLoss()
