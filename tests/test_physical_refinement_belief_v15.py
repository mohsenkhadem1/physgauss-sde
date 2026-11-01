import numpy as np
import pytest
from scripts.physical_refinement_v15.belief import Query, DesignResponses, choose, observe_once


def test_physical_query_ignores_irrelevant_visible_appearance():
    # Independent material and appearance bits: the most visually uncertain
    # observation does not identify the mechanical target.
    material = np.array([0., 0., 1., 1.])[:, None]
    appearance = np.array([0., 100., 0., 100.])[:, None]
    queries = [Query('appearance', 'rgbd', appearance, np.array([1.])),
               Query('poke', 'rgbd_probe', material, np.array([.001]))]
    design = DesignResponses(material, np.ones(1))
    weights = np.ones(4)/4
    assert choose(weights, queries, design, 'visual_uncertainty')[0] == 'appearance'
    assert choose(weights, queries, design, 'physical_risk', fantasies=512)[0] == 'poke'
    posterior, _ = observe_once(weights, queries[1], [1.])
    assert posterior[2:].sum() > .999
    assert np.isclose(posterior[2], posterior[3])


def test_static_identical_views_do_not_identify_hidden_material():
    prior = np.array([.2, .3, .5])
    query = Query('same_exterior', 'rgbd', np.ones((3, 5)), np.ones(5))
    posterior, _ = observe_once(prior, query, np.arange(5.))
    np.testing.assert_allclose(prior, posterior)
    with pytest.raises(ValueError, match='twice'):
        observe_once(prior, query, np.arange(5.), previously_observed=['same_exterior'])


def test_no_reserved_design_or_repeated_query():
    query = Query('q', 'rgbd', np.array([[0.], [1.]]), np.ones(1))
    with pytest.raises(ValueError, match='Evaluation responses'):
        choose([.5, .5], [query], DesignResponses(np.zeros((2, 1)), np.ones(1), role='reserved_evaluation'), 'physical_risk')
    with pytest.raises(ValueError, match='No unobserved'):
        choose([.5, .5], [query], None, 'random', used=['q'])


def test_force_channel_is_not_a_visual_uncertainty_feature():
    q1 = Query('force_rich', 'rgbd_force_probe', np.array([[0., 0.], [0., 100.]]), np.ones(2), visual_indices=(0,))
    q2 = Query('visible', 'rgbd', np.array([[0.], [1.]]), np.ones(1))
    assert choose([.5, .5], [q1, q2], None, 'visual_uncertainty')[0] == 'visible'


def test_correlated_image_evidence_is_not_counted_as_independent_pixels():
    means = np.array([[0., 0.], [1., 1.]])
    correlated = Query('correlated', 'rgbd', means, np.array([[1., .99], [.99, 1.]]))
    independent = Query('independent', 'rgbd', means, np.ones(2))
    a, _ = observe_once([.5, .5], correlated, [1., 1.])
    b, _ = observe_once([.5, .5], independent, [1., 1.])
    assert .5 < a[1] < b[1]
