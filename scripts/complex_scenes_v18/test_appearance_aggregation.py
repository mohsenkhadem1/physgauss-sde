import pytest
from scripts.complex_scenes_v18.score_shared_appearance import aggregate


def test_perfect_initial_image_does_not_inflate_primary_prediction_score():
    rows=[dict(frame=0,psnr=120.),dict(frame=10,psnr=20.),dict(frame=20,psnr=30.)]
    r=aggregate(rows,['psnr'])
    assert r['means']['psnr']==25 and r['means_including_initial_frame']['psnr']==pytest.approx(170/3)
    assert r['noninitial_images']==2 and r['total_images']==3
    with pytest.raises(ValueError,match='noninitial'):aggregate(rows[:1],['psnr'])
