# taken as is from : https://github.com/torchmd/torchmd/blob/master/torchmd/neighbourlist.py

import torch


def discretize_box(box, subcell_size):
    xbins = torch.arange(0, box[0, 0] + subcell_size, subcell_size)
    ybins = torch.arange(0, box[1, 1] + subcell_size, subcell_size)
    zbins = torch.arange(0, box[2, 2] + subcell_size, subcell_size)
    nxbins = len(xbins) - 1
    nybins = len(ybins) - 1
    nzbins = len(zbins) - 1

    r = torch.tensor([-1, 0, 1])
    neighbour_mask = torch.cartesian_prod(r, r, r)

    cellidx = torch.cartesian_prod(
        torch.arange(nxbins), torch.arange(nybins), torch.arange(nzbins)
    )
    cellneighbours = cellidx.unsqueeze(2) + neighbour_mask.T.unsqueeze(0).repeat(
        cellidx.shape[0], 1, 1
    )

    # Can probably be done easier as we only need to handle -1 and max cases, not general -2, max+1 etc
    nbins = torch.tensor([nxbins, nybins, nzbins])[None, :, None].repeat(
        cellidx.shape[0], 1, 27
    )
    negvals = cellneighbours < 0
    cellneighbours[negvals] += nbins[negvals]
    largevals = cellneighbours > (nbins - 1)
    cellneighbours[largevals] -= nbins[largevals]

    return xbins, ybins, zbins, cellneighbours