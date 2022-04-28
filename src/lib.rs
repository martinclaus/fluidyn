pub type Size2D = (usize, usize);
pub type Ix2 = [usize; 2];

pub struct Param {
    g: f64,
    h: f64,
    _rho_0: f64,
}

fn cyclic_shift(idx: usize, shift: isize, len: usize) -> usize {
    let len = len as isize;
    match (idx as isize) + shift {
        i if i < 0 => (i + len) as usize,
        i if i >= len => (i - len) as usize,
        i => i as usize,
    }
}

mod traits {

    use std::ops::{Add, Div, Mul, Sub};

    pub trait Numeric: Copy
    where
        Self: Add<Output = Self>
            + Mul<Output = Self>
            + Mul<Output = Self>
            + Sub<Output = Self>
            + Div<Output = Self>
            + Add<f64, Output = Self>
            + Mul<f64, Output = Self>
            + Mul<f64, Output = Self>
            + Sub<f64, Output = Self>
            + Div<f64, Output = Self>
            + From<f64>
            + Into<f64>
            + Copy,
    {
        fn zero() -> Self;
    }

    impl Numeric for f64 {
        fn zero() -> Self {
            0.0
        }
    }
}

mod field {
    use std::ops::{Add, AddAssign, Index, IndexMut};

    use crate::{Ix2, Size2D};

    pub trait Field<I>
    where
        Self: Sized + Index<Ix2, Output = I> + IndexMut<Ix2, Output = I>,
    {
        fn full(item: I, size: Size2D) -> Self;
        fn size(&self) -> Size2D;
        fn at(&self, idx: [usize; 2]) -> &I;
        fn at_mut(&mut self, idx: [usize; 2]) -> &mut I;
    }
    pub struct Arr2D<I> {
        size: (usize, usize),
        data: Box<[I]>,
    }

    impl<I> Index<[usize; 2]> for Arr2D<I> {
        type Output = I;
        #[inline]
        fn index(&self, index: Ix2) -> &Self::Output {
            &self.data[self.size.1 * index[0] + index[1]]
        }
    }

    impl<I> IndexMut<Ix2> for Arr2D<I> {
        #[inline]
        fn index_mut(&mut self, index: [usize; 2]) -> &mut Self::Output {
            &mut self.data[self.size.1 * index[0] + index[1]]
        }
    }

    impl<I: Add<Output = I> + Copy> Add for Arr2D<I> {
        type Output = Self;

        fn add(self, rhs: Self) -> Self::Output {
            assert_eq!(self.size, rhs.size);
            let data: Box<[I]> = self
                .data
                .iter()
                .zip(rhs.data.iter())
                .map(|(a, b)| *a + *b)
                .collect();
            Self {
                size: self.size,
                data,
            }
        }
    }

    impl<I: AddAssign + Copy> AddAssign<&Arr2D<I>> for Arr2D<I> {
        fn add_assign(&mut self, rhs: &Self) {
            for j in 0..self.size.0 {
                for i in 0..self.size.1 {
                    self[[j, i]] += rhs[[j, i]];
                }
            }
        }
    }

    impl<I: AddAssign + Copy> AddAssign for Arr2D<I> {
        fn add_assign(&mut self, rhs: Self) {
            self.data
                .iter_mut()
                .zip(rhs.data.iter())
                .for_each(|(a, b)| *a += *b);
        }
    }

    impl<I: Copy> Field<I> for Arr2D<I> {
        fn full(item: I, size: Size2D) -> Self {
            Arr2D {
                size,
                data: vec![item; size.0 * size.1].into_boxed_slice(),
            }
        }

        fn size(&self) -> Size2D {
            self.size
        }

        #[inline]
        fn at(&self, idx: [usize; 2]) -> &I {
            &self[idx]
        }

        #[inline]
        fn at_mut(&mut self, idx: [usize; 2]) -> &mut I {
            &mut self[idx]
        }
    }
}

mod mask {
    use std::ops::Mul;

    #[derive(Clone, Copy)]
    pub enum DomainMask {
        Outside,
        Inside,
    }

    pub trait Mask: Copy {
        fn new() -> Self;
        fn inside() -> Self;
        fn outside() -> Self;
        fn is_inside(&self) -> bool;
        fn is_outside(&self) -> bool;
    }

    impl Mask for DomainMask {
        fn new() -> Self {
            DomainMask::Inside
        }

        fn inside() -> Self {
            DomainMask::Inside
        }

        fn outside() -> Self {
            DomainMask::Outside
        }

        fn is_inside(&self) -> bool {
            matches!(self, DomainMask::Inside)
        }

        fn is_outside(&self) -> bool {
            !self.is_inside()
        }
    }

    impl From<DomainMask> for f64 {
        fn from(mask: DomainMask) -> Self {
            match mask {
                DomainMask::Inside => 1f64,
                DomainMask::Outside => 0f64,
            }
        }
    }

    impl Mul<f64> for DomainMask {
        type Output = f64;

        fn mul(self, rhs: f64) -> Self::Output {
            f64::from(self) * rhs
        }
    }
}

mod grid {
    use std::rc::Rc;

    use crate::field::Field;
    use crate::mask::Mask;
    use crate::traits::Numeric;
    use crate::Size2D;

    pub trait Grid<I, M>
    where
        Self: Sized,
        Self::Coord: Field<I>,
        Self::MaskContainer: Field<M>,
        M: Mask,
    {
        type Coord;
        type MaskContainer;

        fn cartesian(size: Size2D, x_start: I, y_start: I, dx: I, dy: I) -> Rc<Self> {
            Rc::new(Self::cartesian_owned(size, x_start, y_start, dx, dy))
        }
        fn cartesian_owned(size: Size2D, x_start: I, y_start: I, dx: I, dy: I) -> Self;
        fn get_x(&self) -> &Self::Coord;
        fn get_dx(&self) -> &Self::Coord;
        fn get_y(&self) -> &Self::Coord;
        fn get_dy(&self) -> &Self::Coord;
        fn get_mask(&self) -> &Self::MaskContainer;
        fn get_mask_mut(&mut self) -> &mut Self::MaskContainer;
        fn with_mask(&mut self, mask: Self::MaskContainer) -> &Self {
            *self.get_mask_mut() = mask;
            self
        }
        fn size(&self) -> Size2D {
            self.get_mask().size()
        }
    }

    pub trait GridTopology<I, M>
    where
        Self::Grid: Grid<I, M>,
        M: Mask,
    {
        type Grid;
        fn get_center(&self) -> Rc<Self::Grid>;

        fn get_h_face(&self) -> Rc<Self::Grid>;

        fn get_v_face(&self) -> Rc<Self::Grid>;

        fn get_corner(&self) -> Rc<Self::Grid>;

        fn cartesian(size: Size2D, x_start: I, y_start: I, dx: I, dy: I) -> Self;

        fn is_v_inside(
            pos: Size2D,
            center_mask: &<Self::Grid as Grid<I, M>>::MaskContainer,
        ) -> bool {
            center_mask[[pos.0, pos.1]].is_inside()
                && center_mask.at([pos.0, pos.1 + 1]).is_inside()
        }
        fn is_h_inside(
            pos: Size2D,
            center_mask: &<Self::Grid as Grid<I, M>>::MaskContainer,
        ) -> bool {
            center_mask.at([pos.0, pos.1]).is_inside()
                && center_mask.at([pos.0 + 1, pos.1 + 1]).is_inside()
        }
        fn is_corner_inside(
            pos: Size2D,
            center_mask: &<Self::Grid as Grid<I, M>>::MaskContainer,
        ) -> bool {
            center_mask.at([pos.0, pos.1]).is_inside()
                && center_mask.at([pos.0, pos.1 + 1]).is_inside()
                && center_mask.at([pos.0 + 1, pos.1 + 1]).is_inside()
                && center_mask.at([pos.0 + 1, pos.1]).is_inside()
        }

        fn make_mask(
            base_mask: &<Self::Grid as Grid<I, M>>::MaskContainer,
            is_inside_strategy: fn(Size2D, &<Self::Grid as Grid<I, M>>::MaskContainer) -> bool,
        ) -> <Self::Grid as Grid<I, M>>::MaskContainer {
            let mut mask =
                <Self::Grid as Grid<I, M>>::MaskContainer::full(M::inside(), base_mask.size());
            for j in 0..base_mask.size().0 {
                for i in 0..base_mask.size().1 {
                    *mask.at_mut([j, i]) = match is_inside_strategy((j, i), base_mask) {
                        true => M::inside(),
                        false => M::outside(),
                    };
                }
            }
            mask
        }
    }
    #[derive(Clone)]
    pub struct Grid2D<CD, CM> {
        x: CD,
        y: CD,
        dx: CD,
        dy: CD,
        mask: CM,
    }

    impl<CD, CM, I, M> Grid<I, M> for Grid2D<CD, CM>
    where
        CD: Field<I>,
        I: Numeric + std::ops::Add,
        CM: Field<M>,
        M: Mask,
    {
        type Coord = CD;
        type MaskContainer = CM;

        fn cartesian_owned(size: Size2D, x_start: I, y_start: I, dx: I, dy: I) -> Self {
            let x = {
                let mut res = CD::full(I::zero(), size);
                (0..size.0).for_each(|j| {
                    (0..size.1).for_each(|i| {
                        *res.at_mut([j, i]) = x_start + dx * (i as f64);
                    })
                });
                res
            };

            let y = {
                let mut res = CD::full(I::zero(), size);
                (0..size.0).for_each(|j| {
                    (0..size.1).for_each(|i| {
                        *res.at_mut([j, i]) = y_start + dy * (j as f64);
                    })
                });
                res
            };

            let mask = {
                let mut res = CM::full(M::inside(), size);
                (0..size.0).for_each(|j| {
                    (0..size.1).for_each(|i| {
                        *res.at_mut([j, i]) =
                            match (j == 0) | (j == size.0 - 1) | (i == 0) | (i == size.1 - 1) {
                                true => M::outside(),
                                false => M::inside(),
                            };
                    })
                });
                res
            };
            Grid2D {
                x,
                y,
                dx: CD::full(dx, size),
                dy: CD::full(dy, size),
                mask,
            }
        }

        fn get_x(&self) -> &Self::Coord {
            &self.x
        }

        fn get_dx(&self) -> &Self::Coord {
            &self.dx
        }

        fn get_y(&self) -> &Self::Coord {
            &self.y
        }

        fn get_dy(&self) -> &Self::Coord {
            &self.dy
        }

        fn get_mask(&self) -> &Self::MaskContainer {
            &self.mask
        }

        fn get_mask_mut(&mut self) -> &mut Self::MaskContainer {
            &mut self.mask
        }
    }

    pub struct StaggeredGrid<G> {
        center: Rc<G>,
        h_side: Rc<G>,
        v_side: Rc<G>,
        corner: Rc<G>,
    }

    impl<I, M, G> GridTopology<I, M> for StaggeredGrid<G>
    where
        G: Grid<I, M>,
        M: Mask,
        I: Numeric,
    {
        type Grid = G;
        fn get_center(&self) -> Rc<G> {
            Rc::clone(&self.center)
        }

        fn get_h_face(&self) -> Rc<G> {
            Rc::clone(&self.h_side)
        }

        fn get_v_face(&self) -> Rc<G> {
            Rc::clone(&self.v_side)
        }

        fn get_corner(&self) -> Rc<G> {
            Rc::clone(&self.corner)
        }

        fn cartesian(size: Size2D, x_start: I, y_start: I, dx: I, dy: I) -> Self {
            let x_shift = x_start + dx * 0.5;
            let y_shift = y_start + dy * 0.5;
            let center = G::cartesian(size, x_start, y_start, dx, dy);

            let mut h_side = G::cartesian_owned(size, x_shift, y_start, dx, dy);
            *h_side.get_mask_mut() = Self::make_mask(center.get_mask(), Self::is_h_inside);
            let h_side = Rc::new(h_side);

            let mut v_side = G::cartesian_owned(size, x_start, y_shift, dx, dy);
            *v_side.get_mask_mut() = Self::make_mask(center.get_mask(), Self::is_v_inside);
            let v_side = Rc::new(v_side);

            let mut corner = G::cartesian_owned(size, x_shift, y_shift, dx, dy);
            *corner.get_mask_mut() = Self::make_mask(center.get_mask(), Self::is_corner_inside);
            let corner = Rc::new(corner);

            StaggeredGrid {
                center,
                h_side,
                v_side,
                corner,
            }
        }
    }
}

mod var {
    use std::ops::{Add, AddAssign};
    use std::rc::Rc;

    use crate::field::Field;
    use crate::grid::Grid;
    use crate::mask::Mask;
    use crate::traits::Numeric;

    pub trait Variable<I, M>
    where
        Self: Sized,
        Self::Data: Field<I>,
        Self::Grid: Grid<I, M>,
        M: Mask,
    {
        type Data;
        type Grid;

        fn zeros(grid: &Rc<Self::Grid>) -> Self;

        fn get_data(&self) -> &Self::Data;

        fn get_data_mut(&mut self) -> &mut Self::Data;

        fn get_grid(&self) -> &Rc<Self::Grid>;
    }
    pub struct Var<CD, G> {
        data: CD,
        grid: Rc<G>,
    }

    impl<CD, G> Add for Var<CD, G>
    where
        CD: Add<Output = CD>,
    {
        type Output = Self;

        fn add(self, rhs: Self) -> Self::Output {
            Self {
                data: self.data + rhs.data,
                grid: Rc::clone(&self.grid),
            }
        }
    }

    impl<CD, G> AddAssign for Var<CD, G>
    where
        CD: AddAssign,
    {
        fn add_assign(&mut self, rhs: Self) {
            self.data += rhs.data;
        }
    }

    impl<CD, G, I, M> Variable<I, M> for Var<CD, G>
    where
        CD: Field<I>,
        G: Grid<I, M>,
        I: Numeric,
        M: Mask,
    {
        type Data = CD;
        type Grid = G;

        fn zeros(grid: &Rc<Self::Grid>) -> Self {
            Self {
                data: Self::Data::full(I::zero(), grid.size()),
                grid: Rc::clone(grid),
            }
        }

        fn get_data(&self) -> &Self::Data {
            &self.data
        }

        fn get_data_mut(&mut self) -> &mut Self::Data {
            &mut self.data
        }

        fn get_grid(&self) -> &Rc<Self::Grid> {
            &self.grid
        }
    }
}

pub mod rhs {
    use crate::{
        cyclic_shift, field::Field, grid::Grid, mask::Mask, traits::Numeric, var::Variable, Param,
    };
    use std::rc::Rc;

    pub fn pg_i_arr<V, I, M>(eta: &V, p: &Param, grid_u: &Rc<V::Grid>) -> V
    where
        V: Variable<I, M>,
        I: Numeric + From<M>,
        M: Mask,
    {
        let dx = eta.get_grid().get_dx();
        let eta_data = eta.get_data();
        let mask_u = grid_u.get_mask();
        let mask_eta = eta.get_grid().get_mask();
        let mut res = V::zeros(grid_u);
        let d = res.get_data_mut();
        let (ny, nx) = d.size();
        let mut ip1: usize;
        for j in 0..ny {
            for i in 0..nx {
                if mask_u.at([j, i]).is_outside() {
                    *d.at_mut([j, i]) = I::zero();
                }
                ip1 = cyclic_shift(i, 1, nx);
                *d.at_mut([j, i]) = (*eta_data.at([j, ip1])
                    * Into::<I>::into(*mask_eta.at([j, ip1]))
                    - *eta_data.at([j, i]) * Into::<I>::into(*mask_eta.at([j, i])))
                    / *dx.at([j, i])
                    * (-p.g);
            }
        }

        res
    }

    pub fn pg_j_arr<V, I, M>(eta: &V, p: &Param, grid_v: &Rc<V::Grid>) -> V
    where
        V: Variable<I, M>,
        I: Numeric + From<M>,
        M: Mask,
    {
        let dy = eta.get_grid().get_dy();
        let eta_data = eta.get_data();
        let mask_eta = eta.get_grid().get_mask();
        let mask_v = grid_v.get_mask();
        let mut res = V::zeros(grid_v);
        let d = res.get_data_mut();
        let mut jp1: usize;
        let (ny, nx) = d.size();
        for j in 0..ny {
            jp1 = cyclic_shift(j, 1, ny);
            for i in 0..nx {
                if mask_v.at([j, i]).is_outside() {
                    continue;
                }
                *d.at_mut([j, i]) = (*eta_data.at([jp1, i])
                    * Into::<I>::into(*mask_eta.at([jp1, i]))
                    - *eta_data.at([j, i]) * Into::<I>::into(*mask_eta.at([j, i])))
                    / *dy.at([j, i])
                    * (-p.g);
            }
        }
        res
    }

    pub fn div_flow<V, I, M>(u: &V, v: &V, p: &Param, grid_eta: &Rc<V::Grid>) -> V
    where
        V: Variable<I, M>,
        I: Numeric + From<M>,
        M: Mask,
    {
        let dx = u.get_grid().get_dx();
        let dy = v.get_grid().get_dy();
        let mask_u = u.get_grid().get_mask();
        let mask_v = v.get_grid().get_mask();
        let mask_eta = grid_eta.get_mask();
        let mut res = V::zeros(grid_eta);
        let d = res.get_data_mut();
        let ud = u.get_data();
        let vd = v.get_data();
        let (ny, nx) = d.size();
        let mut jm1: usize;
        let mut im1: usize;

        for j in 0..ny {
            jm1 = cyclic_shift(j, -1, ny);
            for i in 0..nx {
                if mask_eta.at([j, i]).is_outside() {
                    continue;
                }
                im1 = cyclic_shift(i, -1, nx);
                *d.at_mut([j, i]) = ((Into::<I>::into(*mask_u.at([j, i])) * *ud.at([j, i])
                    - Into::<I>::into(*mask_u.at([j, im1])) * *ud.at([j, im1]))
                    / *dx.at([j, i])
                    + (Into::<I>::into(*mask_v.at([j, i])) * *vd.at([j, i])
                        - Into::<I>::into(*mask_v.at([jm1, i])) * *vd.at([jm1, i]))
                        / *dy.at([j, i]))
                    * (-p.h)
            }
        }
        res
    }
}

pub mod integrate {
    use crate::mask::Mask;
    use crate::state::StateDeque;
    use crate::traits::Numeric;
    use crate::{field::Field, var::Variable};

    const AB2_FAC: [f64; 2] = [-1f64 / 2f64, 3f64 / 2f64];
    const AB3_FAC: [f64; 3] = [5f64 / 12f64, -16f64 / 12f64, 23f64 / 12f64];

    type Integrator<V, I> = fn(&StateDeque<V>, I, [usize; 2]) -> I;

    pub fn time_step<V: Variable<I, M>, I: Numeric, M: Mask>(
        strategy: Integrator<V, I>,
        past_state: &StateDeque<V>,
        step: I,
    ) -> V {
        let mut res = V::zeros(past_state[0].get_grid());
        let d = res.get_data_mut();
        let (ny, nx) = d.size();

        for j in 0..ny {
            for i in 0..nx {
                *d.at_mut([j, i]) = strategy(past_state, step, [j, i]);
            }
        }
        res
    }

    pub fn ef<V: Variable<I, M>, I: Numeric, M: Mask>(
        past_state: &StateDeque<V>,
        step: I,
        idx: [usize; 2],
    ) -> I {
        *past_state[0].get_data().at(idx) * step
    }

    pub fn ab2<V: Variable<I, M>, I: Numeric, M: Mask>(
        past_state: &StateDeque<V>,
        step: I,
        idx: [usize; 2],
    ) -> I {
        if past_state.len() < 2 {
            ef(past_state, step, idx)
        } else {
            (*past_state[0].get_data().at(idx) * AB2_FAC[0]
                + *past_state[1].get_data().at(idx) * AB2_FAC[1])
                * step
        }
    }

    pub fn ab3<V: Variable<I, M>, I: Numeric, M: Mask>(
        past_state: &StateDeque<V>,
        step: I,
        idx: [usize; 2],
    ) -> I {
        if past_state.len() < 3 {
            ab2(past_state, step, idx)
        } else {
            (*past_state[0].get_data().at(idx) * AB3_FAC[0]
                + *past_state[1].get_data().at(idx) * AB3_FAC[1]
                + *past_state[2].get_data().at(idx) * AB3_FAC[2])
                * step
        }
    }
}

mod state {
    use std::{collections::VecDeque, ops::Index};

    pub struct StateDeque<V> {
        inner: VecDeque<V>,
    }

    impl<V> StateDeque<V> {
        pub fn new(capacity: usize) -> StateDeque<V> {
            StateDeque {
                inner: VecDeque::<V>::with_capacity(capacity),
            }
        }
        pub fn push(&mut self, elem: V) {
            if self.inner.len() == self.inner.capacity() {
                self.inner.pop_front();
            }
            self.inner.push_back(elem);
        }

        pub fn len(&self) -> usize {
            self.inner.len()
        }
    }

    impl<V> Index<usize> for StateDeque<V> {
        type Output = V;

        fn index(&self, index: usize) -> &Self::Output {
            self.inner.index(index)
        }
    }
}

#[cfg(test)]
mod tests {
    use std::{error::Error, io::Write, rc::Rc};

    use crate::{
        field::{Arr2D, Field},
        grid::{Grid, Grid2D, GridTopology, StaggeredGrid},
        integrate::{ab3, time_step},
        mask::{DomainMask, Mask},
        rhs::{div_flow, pg_i_arr, pg_j_arr},
        state::StateDeque,
        traits::Numeric,
        var::{Var, Variable},
        Param,
    };

    pub fn rhs<V: Variable<I, M>, I: Numeric + From<M>, M: Mask>(
        u: &V,
        v: &V,
        eta: &V,
        param: &Param,
    ) -> [V; 3] {
        [
            pg_i_arr(eta, param, u.get_grid()),
            pg_j_arr(eta, param, v.get_grid()),
            div_flow(u, v, param, eta.get_grid()),
        ]
    }

    fn eta_init<V, I, M>(grid_eta: &Rc<V::Grid>) -> V
    where
        V: Variable<I, M>,
        I: Numeric,
        M: Mask,
    {
        let mut eta = V::zeros(grid_eta);
        let d = eta.get_data_mut();

        let x = grid_eta.get_x();
        let y = grid_eta.get_y();
        let dx = grid_eta.get_dx();

        let (ny, nx) = d.size();

        let (x_mean, y_mean, dx_mean) = {
            let mut ave_x = 0f64;
            let mut ave_y = 0f64;
            let mut ave_dx = 0f64;
            let n = (x.size().0 * x.size().1) as f64;
            for j in 0..ny {
                for i in 0..nx {
                    ave_x += Into::<f64>::into(*x.at([j, i])) / n;
                    ave_y += Into::<f64>::into(*y.at([j, i])) / n;
                    ave_dx += Into::<f64>::into(*dx.at([j, i])) / n;
                }
            }
            (ave_x, ave_y, ave_dx)
        };

        let var = (dx_mean * (nx as f64) / 5.0).powi(2);
        let mut idx: [usize; 2];
        for j in 0..ny {
            for i in 0..nx {
                idx = [j, i];
                if grid_eta.get_mask().at(idx).is_outside() {
                    continue;
                }
                *d.at_mut(idx) = ((-((Into::<f64>::into(*x.at(idx)) - x_mean).powi(2)
                    + (Into::<f64>::into(*y.at(idx)) - y_mean).powi(2))
                    / var)
                    .exp())
                .into();
            }
        }
        eta
    }

    fn write_to_file<D, I>(file_name: &str, data: &D) -> Result<(), Box<dyn Error>>
    where
        D: Field<I>,
        I: std::fmt::Display,
    {
        let mut file = std::fs::File::create(file_name)?;

        let mut vec = Vec::<String>::with_capacity(data.size().0);
        for j in 0..data.size().0 {
            vec.truncate(0);
            (0..data.size().1).for_each(|i| vec.push(data.at([j, i]).to_string()));
            writeln!(file, "{}", vec.join(", ")).unwrap();
        }
        Ok(())
    }

    #[test]
    fn bench() {
        type D = f64;
        type M = DomainMask;
        type CD = Arr2D<D>;
        type CM = Arr2D<M>;
        type G = Grid2D<CD, CM>;
        type V = Var<CD, G>;

        let param = Param {
            g: 9.80665,
            h: 1.0,
            _rho_0: 1024.0,
        };
        let grid = StaggeredGrid::<G>::cartesian((100, 100), 0.0, 0.0, 1.0, 1.0);
        let mut u = V::zeros(&grid.get_h_face());
        let mut v = V::zeros(&grid.get_v_face());
        let mut eta: V = eta_init(&grid.get_center());
        let mut past_eta = StateDeque::<V>::new(3);
        let mut past_u = StateDeque::<V>::new(3);
        let mut past_v = StateDeque::<V>::new(3);

        let step = 0.05;

        if let Err(err) = write_to_file("eta_init.csv", eta.get_data()) {
            eprintln!("{}", err);
        }

        let now = std::time::Instant::now();

        (0..500).for_each(|_| {
            {
                let [u_inc, v_inc, eta_inc] = rhs(&u, &v, &eta, &param);
                past_u.push(u_inc);
                past_v.push(v_inc);
                past_eta.push(eta_inc);
            }
            u += time_step(ab3, &past_u, step);
            v += time_step(ab3, &past_v, step);
            eta += time_step(ab3, &past_eta, step);
        });
        let t = now.elapsed();

        println!("Time: {:?}", t.as_secs_f64());

        if let Err(err) = write_to_file("eta.csv", eta.get_data()) {
            eprintln!("{}", err);
        }
    }
}
