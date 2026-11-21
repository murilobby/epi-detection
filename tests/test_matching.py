import numpy as np

from epi.eval.matching import Boxes, iou_matrix, match_image, update_confusion

BOX = np.array([[0.0, 0.0, 10.0, 10.0]])


def test_iou_matrix():
    a = np.array([[0.0, 0.0, 2.0, 2.0]])
    b = np.array([[0.0, 0.0, 2.0, 2.0], [1.0, 0.0, 3.0, 2.0], [5.0, 5.0, 6.0, 6.0]])
    np.testing.assert_allclose(iou_matrix(a, b), [[1.0, 1 / 3, 0.0]])


def test_iou_matrix_with_zero_area_boxes():
    point = np.array([[1.0, 1.0, 1.0, 1.0]])
    assert iou_matrix(point, point).tolist() == [[0.0]]


def test_most_confident_detection_takes_the_annotation():
    gt = Boxes(BOX, np.array([2]))
    pred = Boxes(np.array([[0.0, 0.0, 10.0, 9.0], [0.0, 0.0, 10.0, 10.0]]), np.array([2, 2]), np.array([0.9, 0.6]))
    match = match_image(pred, gt, iou_threshold=0.5)
    # A mais confiante leva a anotação, mesmo com IoU menor; a outra fica sem par.
    assert match.pred_to_gt.tolist() == [0, -1]
    assert match.gt_matched.tolist() == [True]
    assert match.confusions == []


def test_detection_of_another_class_is_a_confusion_not_background():
    gt = Boxes(BOX, np.array([2]))
    pred = Boxes(BOX, np.array([1]), np.array([0.8]))
    match = match_image(pred, gt, iou_threshold=0.5)
    assert match.pred_to_gt.tolist() == [-1]
    assert match.gt_matched.tolist() == [False]
    assert match.confusions == [(0, 0)]


def test_update_confusion():
    gt = Boxes(np.array([[0.0, 0.0, 10.0, 10.0], [20.0, 20.0, 30.0, 30.0]]), np.array([0, 1]))
    pred = Boxes(np.array([[0.0, 0.0, 10.0, 10.0], [50.0, 50.0, 60.0, 60.0]]), np.array([0, 1]), np.array([0.9, 0.8]))
    matrix = np.zeros((3, 3), dtype=int)
    update_confusion(matrix, pred, gt, match_image(pred, gt, iou_threshold=0.5))
    # Linhas: classe prevista; colunas: classe verdadeira; o índice 2 é o fundo.
    expected = [
        [1, 0, 0],  # acerto da classe 0
        [0, 0, 1],  # classe 1 detectada sobre o fundo
        [0, 1, 0],  # anotação da classe 1 sem detecção
    ]
    np.testing.assert_array_equal(matrix, expected)
