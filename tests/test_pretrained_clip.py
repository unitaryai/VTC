import clip
import torch

import model.model as module_arch


def test_official_clip_same_as_ours():
    ims = torch.randn(2, 3, 224, 224)
    title = clip.tokenize(["hello", "world"])
    comms = torch.stack(
        [
            clip.tokenize(["hello comm 1", "hello comm 2"]),
            clip.tokenize(["world comm 2", "world comm 2"]),
        ]
    )

    clipmodel, _ = clip.load("ViT-B/32", device="cpu", jit=False)

    ourmodel = module_arch.PretrainedCLIP()
    ourmodel.eval()

    ourmodel_finaltf_skip = module_arch.PretrainedCLIP_finaltf(
        branch_to_adapt_val="skip"
    )
    torch.nn.init.normal_(ourmodel_finaltf_skip.final_linear.weight)
    ourmodel_finaltf_skip.eval()

    clip_im = clipmodel.encode_image(ims)
    clip_txt = clipmodel.encode_text(title)

    our_im, our_txt, _ = ourmodel(ims, title)

    our_im_skiptf, our_txt_skiptf, _ = ourmodel_finaltf_skip(ims, title, comms)

    # Check equal to off-the-shelf clip
    assert torch.allclose(our_im, clip_im / clip_im.norm(dim=-1, keepdim=True))
    assert torch.allclose(our_txt, clip_txt / clip_txt.norm(dim=-1, keepdim=True))

    # Check version with final transformer is identical
    # when final transformer is skipped
    assert torch.allclose(our_im_skiptf, our_im)
    assert torch.allclose(our_txt_skiptf, our_txt)


def test_branch_to_adapt():
    torch.manual_seed(123)

    ims = torch.randn(2, 3, 224, 224)
    title = clip.tokenize(["hello", "world"])
    title2 = clip.tokenize(["goodbye", "world"])
    comms = torch.stack(
        [
            clip.tokenize(["hello comm 1", "hello comm 2"]),
            clip.tokenize(["world comm 2", "world comm 2"]),
        ]
    )

    m_skip = module_arch.PretrainedCLIP_finaltf(branch_to_adapt_val="skip")
    torch.nn.init.normal_(m_skip.final_linear.weight)
    m_vis = module_arch.PretrainedCLIP_finaltf(branch_to_adapt_val="image")
    torch.nn.init.normal_(m_vis.final_linear.weight)
    m_txt = module_arch.PretrainedCLIP_finaltf(branch_to_adapt_val="text")
    torch.nn.init.normal_(m_txt.final_linear.weight)

    m_skip.eval()
    m_vis.eval()
    m_txt.eval()

    imf, titlef, _ = m_skip(ims, title, comms)
    imv, titlev, _ = m_vis(ims, title, comms)
    imt, titlet, _ = m_txt(ims, title, comms)

    # Only the adapted modality should change
    assert torch.allclose(imf, imt)
    assert torch.allclose(titlef, titlev)

    assert not torch.allclose(imv, imf)
    assert not torch.allclose(titlet, titlef)

    # Image feat should stay the same when
    # changing title
    imv2, titlev2, _ = m_vis(ims, title2, comms)

    assert torch.allclose(imv2, imv)
    assert not torch.allclose(titlev2, titlev)
