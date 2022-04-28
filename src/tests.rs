#[test]
fn test_cyclic_shift() {
    use crate::cyclic_shift;
    let size: usize = 10;
    assert_eq!(cyclic_shift(0, 1, size), 1);
    assert_eq!(cyclic_shift(1, 1, size), 2);
    assert_eq!(cyclic_shift(1, -1, size), 0);
    assert_eq!(cyclic_shift(0, -1, size), size - 1);
    assert_eq!(cyclic_shift(size - 1, 2, size), 1);
    assert_eq!(cyclic_shift(1, 3, size), 4);
}
