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
    use crate::{cyclic_shift, Ix2, Size2D};

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

        fn is_v_inside(pos: Ix2, center_mask: &<Self::Grid as Grid<I, M>>::MaskContainer) -> bool {
            let jp1 = cyclic_shift(pos[0], 1, center_mask.size().0);
            center_mask[pos].is_inside() && center_mask[[jp1, pos[1]]].is_inside()
        }
        fn is_h_inside(pos: Ix2, center_mask: &<Self::Grid as Grid<I, M>>::MaskContainer) -> bool {
            let ip1 = cyclic_shift(pos[1], 1, center_mask.size().1);
            center_mask[pos].is_inside() && center_mask[[pos[0], ip1]].is_inside()
        }
        fn is_corner_inside(
            pos: Ix2,
            center_mask: &<Self::Grid as Grid<I, M>>::MaskContainer,
        ) -> bool {
            let ip1 = cyclic_shift(pos[1], 1, center_mask.size().1);
            let jp1 = cyclic_shift(pos[0], 1, center_mask.size().0);
            center_mask[pos].is_inside()
                && center_mask[[pos[0], ip1]].is_inside()
                && center_mask[[jp1, ip1]].is_inside()
                && center_mask[[jp1, pos[1]]].is_inside()
        }

        fn make_mask(
            base_mask: &<Self::Grid as Grid<I, M>>::MaskContainer,
            is_inside_strategy: fn(Ix2, &<Self::Grid as Grid<I, M>>::MaskContainer) -> bool,
        ) -> <Self::Grid as Grid<I, M>>::MaskContainer {
            let mut mask =
                <Self::Grid as Grid<I, M>>::MaskContainer::full(M::inside(), base_mask.size());
            for j in 0..base_mask.size().0 {
                for i in 0..base_mask.size().1 {
                    mask[[j, i]] = match is_inside_strategy([j, i], base_mask) {
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
            Grid2D {
                x: {
                    let mut res = CD::full(I::zero(), size);
                    (0..size.0).for_each(|j| {
                        (0..size.1).for_each(|i| {
                            res[[j, i]] = x_start + dx * (i as f64);
                        })
                    });
                    res
                },
                y: {
                    let mut res = CD::full(I::zero(), size);
                    (0..size.0).for_each(|j| {
                        (0..size.1).for_each(|i| {
                            res[[j, i]] = y_start + dy * (j as f64);
                        })
                    });
                    res
                },
                dx: CD::full(dx, size),
                dy: CD::full(dy, size),
                mask: {
                    let mut res = CM::full(M::inside(), size);
                    (0..size.0).for_each(|j| {
                        (0..size.1).for_each(|i| {
                            res[[j, i]] =
                                match (j == 1) | (j == size.0 - 1) | (i == 0) | (i == size.1 - 1) {
                                    true => M::outside(),
                                    false => M::inside(),
                                };
                        })
                    });
                    res
                },
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

    impl<'a, CD, G> AddAssign<&'a Var<CD, G>> for Var<CD, G>
    where
        CD: AddAssign<&'a CD>,
    {
        fn add_assign(&mut self, rhs: &'a Var<CD, G>) {
            self.data += &rhs.data
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

    pub fn pg_i_arr<V, I, M>(eta: &V, p: &Param, grid_u: &Rc<V::Grid>, pg_i: &mut V)
    where
        V: Variable<I, M>,
        I: Numeric + From<M>,
        M: Mask,
    {
        let dx = eta.get_grid().get_dx();
        let eta_data = eta.get_data();
        let mask_u = grid_u.get_mask();
        let mask_eta = eta.get_grid().get_mask();
        let d = pg_i.get_data_mut();
        let (ny, nx) = d.size();
        let mut ip1: usize;
        for j in 0..ny {
            for i in 0..nx {
                if mask_u[[j, i]].is_outside() {
                    d[[j, i]] = I::zero();
                } else {
                    ip1 = cyclic_shift(i, 1, nx);
                    d[[j, i]] = (eta_data[[j, ip1]] * Into::<I>::into(mask_eta[[j, ip1]])
                        - eta_data[[j, i]] * Into::<I>::into(mask_eta[[j, i]]))
                        / dx[[j, i]]
                        * (-p.g);
                }
            }
        }
    }

    pub fn pg_j_arr<V, I, M>(eta: &V, p: &Param, grid_v: &Rc<V::Grid>, pg_j: &mut V)
    where
        V: Variable<I, M>,
        I: Numeric + From<M>,
        M: Mask,
    {
        let dy = eta.get_grid().get_dy();
        let eta_data = eta.get_data();
        let mask_eta = eta.get_grid().get_mask();
        let mask_v = grid_v.get_mask();
        let d = pg_j.get_data_mut();
        let mut jp1: usize;
        let (ny, nx) = d.size();
        for j in 0..ny {
            jp1 = cyclic_shift(j, 1, ny);
            for i in 0..nx {
                if mask_v[[j, i]].is_outside() {
                    d[[j, i]] = I::zero();
                } else {
                    d[[j, i]] = (eta_data[[jp1, i]] * Into::<I>::into(mask_eta[[jp1, i]])
                        - eta_data[[j, i]] * Into::<I>::into(mask_eta[[j, i]]))
                        / dy[[j, i]]
                        * (-p.g);
                }
            }
        }
    }

    pub fn div_flow<V, I, M>(u: &V, v: &V, p: &Param, grid_eta: &Rc<V::Grid>, div: &mut V)
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
        let d = div.get_data_mut();
        let ud = u.get_data();
        let vd = v.get_data();
        let (ny, nx) = d.size();
        let mut jm1: usize;
        let mut im1: usize;

        for j in 0..ny {
            jm1 = cyclic_shift(j, -1, ny);
            for i in 0..nx {
                if mask_eta[[j, i]].is_outside() {
                    d[[j, i]] = I::zero();
                } else {
                    im1 = cyclic_shift(i, -1, nx);
                    d[[j, i]] = ((Into::<I>::into(mask_u[[j, i]]) * ud[[j, i]]
                        - Into::<I>::into(mask_u[[j, im1]]) * ud[[j, im1]])
                        / dx[[j, i]]
                        + (Into::<I>::into(mask_v[[j, i]]) * vd[[j, i]]
                            - Into::<I>::into(mask_v[[jm1, i]]) * vd[[jm1, i]])
                            / dy[[j, i]])
                        * (-p.h)
                }
            }
        }
    }
}

pub mod integrate {
    use std::marker::PhantomData;

    use crate::mask::Mask;
    use crate::state::StateDeque;
    use crate::traits::Numeric;
    use crate::{field::Field, var::Variable};

    const AB2_FAC: [f64; 2] = [-1f64 / 2f64, 3f64 / 2f64];
    const AB3_FAC: [f64; 3] = [5f64 / 12f64, -16f64 / 12f64, 23f64 / 12f64];

    pub trait Integrator<V, I, M> {
        fn call(past_state: &StateDeque<&V>, step: I, idx: [usize; 2]) -> I;
    }

    pub struct Integrate<I> {
        _integrator_type: PhantomData<I>,
    }

    impl<IS> Integrate<IS> {
        pub fn compute_inc<V: Variable<I, M>, I, M: Mask>(
            past_state: StateDeque<&V>,
            step: I,
            out: &mut V,
        ) where
            IS: Integrator<V, I, M>,
            V: Variable<I, M>,
            I: Copy,
            M: Mask,
        {
            let d = out.get_data_mut();
            let (ny, nx) = d.size();

            for j in 0..ny {
                for i in 0..nx {
                    d[[j, i]] = IS::call(&past_state, step, [j, i]);
                }
            }
        }
    }

    pub struct Ef;

    impl<V, I, M> Integrator<V, I, M> for Ef
    where
        V: Variable<I, M>,
        M: Mask,
        I: Numeric,
    {
        fn call(past_state: &StateDeque<&V>, step: I, idx: [usize; 2]) -> I {
            past_state[0].get_data()[idx] * step
        }
    }

    pub struct Ab2;

    impl<V, I, M> Integrator<V, I, M> for Ab2
    where
        V: Variable<I, M>,
        M: Mask,
        I: Numeric,
    {
        fn call(past_state: &StateDeque<&V>, step: I, idx: [usize; 2]) -> I {
            if past_state.len() < 2 {
                Ef::call(past_state, step, idx)
            } else {
                (past_state[0].get_data()[idx] * AB2_FAC[0]
                    + past_state[1].get_data()[idx] * AB2_FAC[1])
                    * step
            }
        }
    }

    pub struct Ab3;

    impl<V, I, M> Integrator<V, I, M> for Ab3
    where
        V: Variable<I, M>,
        M: Mask,
        I: Numeric,
    {
        fn call(past_state: &StateDeque<&V>, step: I, idx: [usize; 2]) -> I {
            if past_state.len() < 3 {
                Ab2::call(past_state, step, idx)
            } else {
                (past_state[0].get_data()[idx] * AB3_FAC[0]
                    + past_state[1].get_data()[idx] * AB3_FAC[1]
                    + past_state[2].get_data()[idx] * AB3_FAC[2])
                    * step
            }
        }
    }
}

pub mod state {
    use std::{
        collections::{HashMap, VecDeque},
        hash::Hash,
        ops::{Index, IndexMut},
        rc::Rc,
    };

    use crate::{mask::Mask, var::Variable};

    #[derive(Copy, Clone, PartialEq, Eq, Hash)]
    pub enum SWMVars {
        U,
        V,
        ETA,
    }

    pub trait VarSet: Sized + Copy {
        fn values() -> &'static [Self];
    }

    impl VarSet for SWMVars {
        fn values() -> &'static [Self] {
            &[Self::U, Self::V, Self::ETA][..]
        }
    }

    pub trait StateContainer<K, V>
    where
        Self: Index<K, Output = V> + IndexMut<K, Output = V>,
    {
        fn new<I, M>(grid_map: &[GridMap<K, <V as Variable<I, M>>::Grid>]) -> Self
        where
            V: Variable<I, M>,
            M: Mask;
    }

    pub struct State<K, V> {
        vars: HashMap<K, V>,
    }

    type GridMap<K, G> = (K, Rc<G>);

    impl<K, V> StateContainer<K, V> for State<K, V>
    where
        K: Eq + Hash + VarSet,
    {
        fn new<I, M>(grid_map: &[GridMap<K, <V as Variable<I, M>>::Grid>]) -> Self
        where
            V: Variable<I, M>,
            M: Mask,
        {
            let mut res = Self {
                vars: HashMap::new(),
            };

            for (k, g) in grid_map {
                res.vars.insert(*k, V::zeros(g));
            }
            res
        }
    }

    impl<K, V> Index<K> for State<K, V>
    where
        K: Eq + Hash,
    {
        type Output = V;

        fn index(&self, index: K) -> &Self::Output {
            self.vars.get(&index).unwrap()
        }
    }

    impl<K, V> IndexMut<K> for State<K, V>
    where
        K: Eq + Hash,
    {
        fn index_mut(&mut self, index: K) -> &mut Self::Output {
            self.vars.get_mut(&index).unwrap()
        }
    }

    pub trait StateBuilder<K, V, I, M>
    where
        V: Variable<I, M>,
        M: Mask,
    {
        fn new(grid_map: &[GridMap<K, V::Grid>]) -> Self;
        fn get_grid_map(&self) -> &[GridMap<K, V::Grid>];

        fn make<S: StateContainer<K, V>>(&self) -> S {
            S::new(self.get_grid_map())
        }
    }
    pub struct Builder<K, G> {
        grid_map: Box<[GridMap<K, G>]>,
    }

    impl<K, V, I, M> StateBuilder<K, V, I, M> for Builder<K, V::Grid>
    where
        V: Variable<I, M>,
        M: Mask,
        K: Clone,
    {
        fn make<S: StateContainer<K, V>>(&self) -> S {
            S::new(self.grid_map.as_ref())
        }

        fn new(grid_map: &[GridMap<K, V::Grid>]) -> Self {
            Self {
                grid_map: grid_map.to_vec().into_boxed_slice(),
            }
        }

        fn get_grid_map(&self) -> &[GridMap<K, V::Grid>] {
            self.grid_map.as_ref()
        }
    }

    pub trait StateProvider<SC, SB, K, V, I, M>
    where
        SB: StateBuilder<K, V, I, M>,
        SC: StateContainer<K, V>,
        V: Variable<I, M>,
        M: Mask,
    {
        fn new(builder: SB, capacity: usize) -> Self;

        fn get(&mut self) -> SC;

        fn take(&mut self, state: SC);
    }

    pub struct Provider<SB, SC> {
        builder: SB,
        state_buffer: VecDeque<SC>,
    }

    impl<SC, SB, K, V, I, M> StateProvider<SC, SB, K, V, I, M> for Provider<SB, SC>
    where
        SB: StateBuilder<K, V, I, M>,
        SC: StateContainer<K, V>,
        V: Variable<I, M>,
        M: Mask,
    {
        fn new(builder: SB, capacity: usize) -> Self {
            Self {
                builder,
                state_buffer: VecDeque::with_capacity(capacity),
            }
        }

        fn get(&mut self) -> SC {
            if self.state_buffer.is_empty() {
                self.builder.make()
            } else {
                self.state_buffer.pop_front().unwrap()
            }
        }

        fn take(&mut self, state: SC) {
            self.state_buffer.push_back(state);
        }
    }

    pub struct StateDeque<S> {
        inner: VecDeque<S>,
    }

    impl<S> StateDeque<S> {
        pub fn new(capacity: usize) -> StateDeque<S> {
            StateDeque {
                inner: VecDeque::<S>::with_capacity(capacity),
            }
        }

        pub fn push(&mut self, elem: S) -> Option<S> {
            let res = match self.inner.len() == self.inner.capacity() {
                true => self.inner.pop_front(),
                false => None,
            };

            self.inner.push_back(elem);
            res
        }

        pub fn len(&self) -> usize {
            self.inner.len()
        }

        pub fn is_empty(&self) -> bool {
            self.inner.is_empty()
        }

        pub fn take_var<K, V, I, M>(&self, var: K) -> StateDeque<&V>
        where
            S: StateContainer<K, V>,
            K: VarSet,
            V: Variable<I, M>,
            M: Mask,
        {
            StateDeque {
                inner: { self.inner.iter().map(|s| &s[var]).collect() },
            }
        }
    }

    impl<V> Index<usize> for StateDeque<V> {
        type Output = V;

        fn index(&self, index: usize) -> &Self::Output {
            self.inner.index(index)
        }
    }

    impl<V> IndexMut<usize> for StateDeque<V> {
        fn index_mut(&mut self, index: usize) -> &mut Self::Output {
            self.inner.index_mut(index)
        }
    }
}

#[cfg(test)]
mod benchmark {
    use std::{error::Error, io::Write, rc::Rc};

    use crate::{
        field::{Arr2D, Field},
        grid::{Grid, Grid2D, GridTopology, StaggeredGrid},
        integrate::{Ab3, Integrate},
        mask::{DomainMask, Mask},
        rhs::{div_flow, pg_i_arr, pg_j_arr},
        state,
        state::StateDeque,
        state::{SWMVars, State, StateBuilder, StateContainer, StateProvider},
        traits::Numeric,
        var::{Var, Variable},
        Param,
    };

    pub fn rhs<S: StateContainer<SWMVars, V>, V: Variable<I, M>, I: Numeric + From<M>, M: Mask>(
        state: &S,
        param: &Param,
        mut new_state: S,
    ) -> S {
        pg_i_arr(
            &state[SWMVars::ETA],
            param,
            state[SWMVars::U].get_grid(),
            &mut new_state[SWMVars::U],
        );
        pg_j_arr(
            &state[SWMVars::ETA],
            param,
            state[SWMVars::V].get_grid(),
            &mut new_state[SWMVars::V],
        );
        div_flow(
            &state[SWMVars::U],
            &state[SWMVars::V],
            param,
            state[SWMVars::ETA].get_grid(),
            &mut new_state[SWMVars::ETA],
        );
        new_state
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
                    ave_x += Into::<f64>::into(x[[j, i]]) / n;
                    ave_y += Into::<f64>::into(y[[j, i]]) / n;
                    ave_dx += Into::<f64>::into(dx[[j, i]]) / n;
                }
            }
            (ave_x, ave_y, ave_dx)
        };

        let var = (dx_mean * (nx as f64) / 5.0).powi(2);
        let mut idx: [usize; 2];
        for j in 0..ny {
            for i in 0..nx {
                idx = [j, i];
                if grid_eta.get_mask()[idx].is_outside() {
                    continue;
                }
                d[idx] = ((-((Into::<f64>::into(x[idx]) - x_mean).powi(2)
                    + (Into::<f64>::into(y[idx]) - y_mean).powi(2))
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
            (0..data.size().1).for_each(|i| vec.push(data[[j, i]].to_string()));
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
        let state_builder = {
            let grid = StaggeredGrid::<G>::cartesian((100, 100), 0.0, 0.0, 1.0, 1.0);
            let grid_map = vec![
                (SWMVars::U, grid.get_h_face()),
                (SWMVars::V, grid.get_v_face()),
                (SWMVars::ETA, grid.get_center()),
            ];
            <state::Builder<_, _> as StateBuilder<_, V, _, _>>::new(&grid_map)
        };
        let mut state_provider = state::Provider::<_, State<_, V>>::new(state_builder, 10);
        let mut state = state_provider.get();
        state[SWMVars::ETA] = eta_init(state[SWMVars::ETA].get_grid());
        let mut rhs_evals = StateDeque::new(3);

        let step = 0.05;

        if let Err(err) = write_to_file("eta_init.csv", state[SWMVars::ETA].get_data()) {
            eprintln!("{}", err);
        }

        let now = std::time::Instant::now();

        (0..500).for_each(|_| {
            match rhs_evals.push(rhs(&state, &param, state_provider.get())) {
                Some(state) => state_provider.take(state),
                None => (),
            };
            let mut inc = state_provider.get();

            Integrate::<Ab3>::compute_inc(
                rhs_evals.take_var(SWMVars::U),
                step,
                &mut inc[SWMVars::U],
            );
            Integrate::<Ab3>::compute_inc(
                rhs_evals.take_var(SWMVars::V),
                step,
                &mut inc[SWMVars::V],
            );
            Integrate::<Ab3>::compute_inc(
                rhs_evals.take_var(SWMVars::ETA),
                step,
                &mut inc[SWMVars::ETA],
            );

            state[SWMVars::U] += &inc[SWMVars::U];
            state[SWMVars::V] += &inc[SWMVars::V];
            state[SWMVars::ETA] += &inc[SWMVars::ETA];
        });
        let t = now.elapsed();

        println!("Time: {:?}", t.as_secs_f64());

        if let Err(err) = write_to_file("eta.csv", state[SWMVars::ETA].get_data()) {
            eprintln!("{}", err);
        }
        if let Err(err) = write_to_file("u.csv", state[SWMVars::U].get_data()) {
            eprintln!("{}", err);
        }
        if let Err(err) = write_to_file("v.csv", state[SWMVars::V].get_data()) {
            eprintln!("{}", err);
        }
    }
}

#[cfg(test)]
mod tests;
