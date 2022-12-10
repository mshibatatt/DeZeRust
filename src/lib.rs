pub mod DeZeRust {
    use std::collections::{BinaryHeap, HashSet};
    use std::rc::{Rc, Weak};
    use std::cell::RefCell;
    use std::cmp::{Ord, Ordering};

    type Dtype = f64;
    type Grad = Option<Variable>;
    static mut ENABLE_BACKPROP: bool = true;
    static mut RETAIN_GRAD: bool = true;
    static mut CREATE_SECOND_GRAPH: bool = true;

    unsafe fn config_backprop_on() {
        ENABLE_BACKPROP = true;
    }
    unsafe fn config_backprop_off() {
        ENABLE_BACKPROP = false;
    }
    unsafe fn config_retain_on() {
        ENABLE_BACKPROP = true;
    }
    unsafe fn config_retain_off() {
        ENABLE_BACKPROP = false;
    }
    unsafe fn config_second_graph_on() {
        CREATE_SECOND_GRAPH = true;
    }
    unsafe fn config_second_graph_off() {
        CREATE_SECOND_GRAPH = false;
    }
    
    #[derive(Clone)]
    pub struct Variable_ {
        data: Dtype,
        name: String,
        grad: Grad,
        creator: Option<Function>,
        generation: usize,
    }

    impl Variable_ {
        fn new(x: Dtype) -> Self {
            Variable_ {
                data: x,
                name: String::new(),
                grad: None,
                creator: None,
                generation: 0,
            }
        }

        pub fn set_creator(&mut self, f: Function) {
            self.creator = Some(f.clone());
            self.generation = f.0.borrow().generation + 1;
        }

        pub fn set_name<S: Into<String>>(&mut self, name: S) {
            self.name = name.into();
        }

        pub fn clear_grad(&mut self) {
            self.grad = None;
        }
    }

    #[derive(Clone)]
    pub struct Variable(Rc<RefCell<Variable_>>);

    impl Variable {
        pub fn new(x: Dtype) -> Self {
            let v = Variable_::new(x);
            Variable(Rc::new(RefCell::new(v)))
        }

        fn set_creator(&self, f: Function) {
            self.0.as_ref().borrow_mut().set_creator(f);
        }

        pub fn set_name<S: Into<String>>(&self, name: S) {
            self.0.as_ref().borrow_mut().set_name(name);
        }

        pub fn clear_grad(&self) {
            self.0.as_ref().borrow_mut().grad = None;
        }

        pub fn backward(&self) {
            if let None = self.get_grad_data() {
                self.update_grad(Some(Variable::new(1.0)));
            }

            let mut seen = HashSet::new();
            let mut que = BinaryHeap::new();

            let v = self.clone();
            let creator = &v.0.borrow().creator;
            if let Some(f) = creator {
                let listed_f = f.clone();
                que.push(listed_f);
                let seen_f = f.clone();
                seen.insert(seen_f.0.as_ptr());
            }
            while !que.is_empty() {
                let f = que.pop().unwrap();

                let gys: Vec<Grad> = f.get_output_grad();

                if unsafe { !CREATE_SECOND_GRAPH } & unsafe { ENABLE_BACKPROP } {
                    unsafe { config_backprop_off(); } 
                }

                let gxs: Vec<Grad> = f.backward(&gys);
                let x_back = &f.0.borrow().input;

                for (x, gx) in x_back.iter().zip(gxs) {
                    x.update_grad(gx);
                    // println!("{}: {:?}", x.get_name(), x.get_grad_data());
                    let x_temp = x.0.borrow();
                    let nxt_creator = x_temp.creator.as_ref(); 
                    if let Some(g) = nxt_creator {
                        if !seen.contains(&g.0.as_ptr()) {
                            let listed_g = g.clone();
                            que.push(listed_g);
                            let seen_g = g.clone();
                            seen.insert(seen_g.0.as_ptr());
                        }
                    }
                }

                if unsafe { CREATE_SECOND_GRAPH } & unsafe { !ENABLE_BACKPROP } {
                    unsafe { config_backprop_on(); } 
               }

                if unsafe { !RETAIN_GRAD } {
                    // TODO: Fix runtime error in borrow mut
                    for y in f.0.borrow().output.iter() {
                        y.upgrade().unwrap().as_ref().borrow_mut().clear_grad();
                    }
                }
            }
        }

        pub fn update_grad(&self, g: Grad) {
            match self.get_grad() {
                None => self.0.as_ref().borrow_mut().grad = g,
                Some(d) => self.0.as_ref().borrow_mut().grad = Some(d.get_data() + g.unwrap()),
            }
        }

        pub fn get_data(&self) -> Dtype {
            self.0.borrow().data
        }

        pub fn get_grad(&self) -> Grad {
            self.0.borrow().grad.clone()
        }

        pub fn get_grad_data(&self) -> Option<Dtype> {
            match self.0.borrow().grad.as_ref() {
                None => None,
                Some(g) => Some(g.get_data()),
            }
        }

        pub fn get_name(&self) -> String {
            self.0.borrow().name.to_owned()
        }

        pub fn square(self) -> Self {
            square(self)
        }

        pub fn exp(self) -> Self {
            exp(self)
        }

    }


    #[derive(Clone)]
    enum FunctionTypes {
        None,
        ADD,
        MUL,
        SUB,
        DIV,
        EXP,
        SQUARE,
        POW,
        NEG,
        SIGMOID,
        RELU,
        SOFTMAX,
        SUM,
        DOT,
        MSE,
    }

    #[derive(Clone)]
    pub struct Function_ {
        ftype: FunctionTypes,
        input: Vec<Variable>,
        output: Vec<Weak<RefCell<Variable_>>>,
        generation: usize,
    }

    #[derive(Clone)]
    pub struct Function(Rc<RefCell<Function_>>);

    impl Function {
        fn call(ftype: FunctionTypes, input: &Vec<Variable>) -> Vec<Variable> {
            let x = input.iter().map(|v| v.get_data()).collect::<Vec<Dtype>>();
            let y = _forward(&ftype, &x);
            let output = y.iter().map(|v| Variable::new(*v)).collect::<Vec<Variable>>();

            if unsafe { ENABLE_BACKPROP } {
                let f = Function_ { 
                    ftype,
                    input: input.clone(), 
                    output: output.iter().map(|v| Rc::downgrade(&v.0)).collect(),
                    generation: input.iter().map(|v| v.0.borrow().generation).max().unwrap(),
                };

                // set creator in output Variables
                let box_f = Rc::new(RefCell::new(f.clone()));
                for v in f.output.iter() {
                    v.upgrade().unwrap().as_ref().borrow_mut().set_creator(Function(Rc::clone(&box_f)));
                }
                output
            } else {
                output
            }
        }

        fn forward(&self, x: &Vec<Dtype>) -> Vec<Dtype> {
            let ftype = &self.0.borrow().ftype;
            _forward(&ftype, &x)
        }

        fn backward(&self, dy: &Vec<Grad>) -> Vec<Grad> {
            let x: Vec<Variable> = self.0.borrow().input.iter().map(|v| v.clone()).collect();
            match self.0.borrow().ftype {
                FunctionTypes::ADD => {
                    assert_eq!(x.len(), 2);
                    assert_eq!(dy.len(), 1);
                    let d = dy[0].clone().unwrap();
                    vec![Some(d.clone()), Some(d.clone())]
                },
                FunctionTypes::MUL => {
                    assert_eq!(x.len(), 2);
                    assert_eq!(dy.len(), 1);
                    let d = dy[0].clone().unwrap();
                    vec![Some(d.clone()*x[1].clone()), Some(d.clone()*x[0].clone())]
                },
                FunctionTypes::SUB => {
                    assert_eq!(x.len(), 2);
                    assert_eq!(dy.len(), 1);
                    let d = dy[0].clone().unwrap();
                    vec![Some(d.clone()), Some(-d.clone())]
                },
                FunctionTypes::DIV => {
                    assert_eq!(x.len(), 2);
                    assert_eq!(dy.len(), 1);
                    let d = dy[0].clone().unwrap();
                    vec![Some(d.clone()/x[1].clone()), Some(-d.clone()*x[0].clone()/(x[1].clone()*x[1].clone()))]
                },
                FunctionTypes::SQUARE => {
                    assert_eq!(x.len(), 1);
                    assert_eq!(dy.len(), 1);
                    let d = dy[0].clone().unwrap();
                    vec![Some(2.0 * d.clone() * x[0].clone())]
                },
                FunctionTypes::EXP => {
                    assert_eq!(x.len(), 1);
                    assert_eq!(dy.len(), 1);
                    let d = dy[0].clone().unwrap();
                    vec![Some(d.clone() * x[0].clone().exp())]
                },
                FunctionTypes::NEG => {
                    assert_eq!(x.len(), 1);
                    assert_eq!(dy.len(), 1);
                    let d = dy[0].clone().unwrap();
                    vec![Some(-d.clone())]
                },
                FunctionTypes::SUM => {
                    assert_eq!(dy.len(), 1);
                    let d = dy[0].clone().unwrap();
                    vec![Some(-d.clone()); x.len()]
                },
                FunctionTypes::DOT => {
                    assert_eq!(x.len() % 2, 0);
                    assert_eq!(x.len() / 2, dy.len());
                    let center = x.len() / 2 as usize;
                    let mut res: Vec<Grad> = vec![];
                    for i in center..x.len() {
                        res.push(Some(dy[i].clone().unwrap()));
                    }
                    for i in 0..center {
                        res.push(Some(dy[i].clone().unwrap()));
                    }
                    res
                },
                FunctionTypes::MSE => {
                    assert_eq!(dy.len(), 1);
                    assert_eq!(x.len() % 2, 0);
                    let center = x.len() / 2 as usize;
                    let mut res: Vec<Grad> = vec![];
                    for i in 0..x.len() {
                        if i < center {
                            let diff: Variable = x[i].clone() - x[i+center].clone();
                            res.push(Some(dy[0].clone().unwrap() * diff * (2.0 / center as Dtype)));
                        } else {
                            res.push(Some(-res[i-center].clone().unwrap()));
                        }
                    }
                    res
                },
                _ => unimplemented!(),
            }
        }

        fn get_output_grad(&self) -> Vec<Grad> {
            self.0.borrow().output.iter().map(|p| p.upgrade().unwrap().borrow().grad.clone()).collect()
        }
    }

    impl Ord for Function {
        fn cmp(&self, other: &Self) -> Ordering {
            self.0.borrow().generation.cmp(&other.0.borrow().generation)
        }
    }
    
    impl PartialOrd for Function {
        fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
            Some(self.cmp(other))
        }
    }
    
    impl PartialEq for Function {
        fn eq(&self, other: &Self) -> bool {
            self.0.borrow().generation == other.0.borrow().generation
        }
    }

    impl Eq for Function {}


    fn _forward(ftype: &FunctionTypes, x: &Vec<Dtype>) -> Vec<Dtype> {
        match ftype {
            FunctionTypes::ADD => {
                assert_eq!(x.len(), 2);
                vec![x[0] + x[1]]
            },
            FunctionTypes::MUL => {
                assert_eq!(x.len(), 2);
                vec![x[0] * x[1]]
            },
            FunctionTypes::SUB => {
                assert_eq!(x.len(), 2);
                vec![x[0] - x[1]]
            },
            FunctionTypes::DIV => {
                assert_eq!(x.len(), 2);
                vec![x[0] / x[1]]
            },
            FunctionTypes::SQUARE => {
                assert_eq!(x.len(), 1);
                vec![x[0] * x[0]]
            },
            FunctionTypes::EXP => {
                assert_eq!(x.len(), 1);
                vec![x[0].exp()]
            },
            FunctionTypes::NEG => {
                assert_eq!(x.len(), 1);
                vec![-x[0]]
            },
            FunctionTypes::SUM => {
                vec![x.iter().sum()]
            },
            FunctionTypes::DOT => {
                assert_eq!(x.len() % 2, 0);
                let center = x.len() / 2 as usize;
                let mut res = vec![];
                for i in 0..center {
                    res.push(x[i] * x[i+center]);
                }
                res
            },
            FunctionTypes::MSE => {
                assert_eq!(x.len() % 2, 0);
                let center = x.len() / 2 as usize;
                let mut res: Dtype = 0.0;
                for i in 0..center {
                    res += (x[i] - x[i+center])*(x[i] - x[i+center]);
                }
                vec![res / x.len() as Dtype]
            },
            
            _ => unimplemented!()
        }
    }

    pub fn square(input: Variable) -> Variable {
        Function::call(FunctionTypes::SQUARE, &vec![input])[0].clone()
    }

    pub fn exp(input: Variable) -> Variable {
        Function::call(FunctionTypes::EXP, &vec![input])[0].clone()
    }

    pub fn add(input_0: Variable, input_1: Variable) -> Variable {
        Function::call(FunctionTypes::ADD, &vec![input_0, input_1])[0].clone()
    }

    pub fn mul(input_0: Variable, input_1: Variable) -> Variable {
        Function::call(FunctionTypes::MUL, &vec![input_0, input_1])[0].clone()
    }

    pub fn sub(input_0: Variable, input_1: Variable) -> Variable {
        Function::call(FunctionTypes::SUB, &vec![input_0, input_1])[0].clone()
    }

    pub fn div(input_0: Variable, input_1: Variable) -> Variable {
        Function::call(FunctionTypes::DIV, &vec![input_0, input_1])[0].clone()
    }

    pub fn neg(input: Variable) -> Variable {
        Function::call(FunctionTypes::NEG, &vec![input])[0].clone()
    }

    pub fn sum(input: &Vec<Variable>) -> Variable {
        Function::call(FunctionTypes::SUM, input)[0].clone()
    }

    pub fn dot(input_0: &Vec<Variable>, input_1: &Vec<Variable>) -> Vec<Variable> {
        let mut input = input_0.clone();
        for i in input_1.iter() {
            input.push(i.clone())
        }
        Function::call(FunctionTypes::DOT, &input)
    }

    pub fn mean_squared_error(input_0: &Vec<Variable>, input_1: &Vec<Variable>) -> Variable {
        let mut input = input_0.clone();
        for i in input_1.iter() {
            input.push(i.clone())
        }
        Function::call(FunctionTypes::MSE, &input)[0].clone()
    }


    impl std::ops::Add<Variable> for Variable {
        type Output = Variable;
        fn add(self, _rhs: Variable) -> Variable {
            add(self, _rhs)
        }
    }

    impl std::ops::Add<Dtype> for Variable {
        type Output = Variable;
        fn add(self, _rhs: Dtype) -> Variable {
            add(self, Variable::new(_rhs))
        }
    }

    impl std::ops::Add<Variable> for Dtype {
        type Output = Variable;
        fn add(self, _rhs: Variable) -> Variable {
            add(Variable::new(self), _rhs)
        }
    }

    impl std::ops::Mul<Variable> for Variable {
        type Output = Variable;
        fn mul(self, _rhs: Variable) -> Variable {
            mul(self, _rhs)
        }
    }

    impl std::ops::Mul<Dtype> for Variable {
        type Output = Variable;
        fn mul(self, _rhs: Dtype) -> Variable {
            mul(self, Variable::new(_rhs))
        }
    }

    impl std::ops::Mul<Variable> for Dtype {
        type Output = Variable;
        fn mul(self, _rhs: Variable) -> Variable {
            mul(Variable::new(self), _rhs)
        }
    }

    impl std::ops::Sub<Variable> for Variable {
        type Output = Variable;
        fn sub(self, _rhs: Variable) -> Variable {
            sub(self, _rhs)
        }
    }

    impl std::ops::Sub<Dtype> for Variable {
        type Output = Variable;
        fn sub(self, _rhs: Dtype) -> Variable {
            sub(self, Variable::new(_rhs))
        }
    }

    impl std::ops::Sub<Variable> for Dtype {
        type Output = Variable;
        fn sub(self, _rhs: Variable) -> Variable {
            sub(Variable::new(self), _rhs)
        }
    }

    impl std::ops::Div<Variable> for Variable {
        type Output = Variable;
        fn div(self, _rhs: Variable) -> Variable {
            div(self, _rhs)
        }
    }

    impl std::ops::Div<Dtype> for Variable {
        type Output = Variable;
        fn div(self, _rhs: Dtype) -> Variable {
            div(self, Variable::new(_rhs))
        }
    }

    impl std::ops::Div<Variable> for Dtype {
        type Output = Variable;
        fn div(self, _rhs: Variable) -> Variable {
            div(Variable::new(self), _rhs)
        }
    }

    impl std::ops::Neg for Variable {
        type Output = Variable;
        fn neg(self) -> Variable {
            neg(self)
        }
    }

    pub fn numerical_diff(f: &dyn Fn(Dtype) -> Dtype, x: Dtype) -> Dtype {
        let eps = 1e-6;
        (f(x + eps) - f(x - eps)) / (2.0*eps)
    }

    #[cfg(test)]
    mod test {
        use crate::DeZeRust::*;
        
        #[test]
        fn test_backward_8() {
            // unsafe { config_retain_on() };
            let x = Variable::new(3.0);
            let y = square(x.clone());
            let z = square(y.clone());
            z.backward();
            assert_eq!(x.get_grad_data(), Some(108.0));
            assert_eq!(y.get_grad_data(), Some(18.0));
            // unsafe { config_retain_off() };
        }

        #[test]
        fn test_backward_13() {
            // unsafe { config_retain_on() };
            let (x, y) = (Variable::new(2.0), Variable::new(3.0));
            let z = add(square(x.clone()), square(y.clone()));
            z.backward();
            assert_eq!(z.get_data(), 13.0);
            assert_eq!(x.get_grad_data(), Some(4.0));
            assert_eq!(y.get_grad_data(), Some(6.0));
            // unsafe { config_retain_off() };
        }

        #[test]
        fn test_backward_14() {
            let x = Variable::new(3.0);
            let y = add(x.clone(), x.clone());
            y.backward();
            assert_eq!(x.get_grad_data(), Some(2.0));

            x.clear_grad();
            let y = add(add(x.clone(), x.clone()).clone(), x.clone());
            y.backward();
            assert_eq!(x.get_grad_data(), Some(3.0));
        }

        #[test]
        fn test_backward_16() {
            let x = Variable::new(2.0);
            let a = square(x.clone()).clone();
            let y = add(square(a.clone()).clone(), square(a.clone()).clone());
            y.backward();
            assert_eq!(y.get_data(), 32.0);
            assert_eq!(x.get_grad_data(), Some(64.0));
        }

        #[test]
        fn test_goldstein_price() {
            fn goldstein(x: Variable, y: Variable) -> Variable {
                (1.0 + (x.clone() + y.clone() + 1.0).square() * 
                    (19.0 - 14.0*x.clone() + 3.0*x.clone().square() - 
                        14.0*y.clone() + 6.0*x.clone()*y.clone() + 3.0*y.clone().square())) *

                (30.0 + (2.0*x.clone() - 3.0*y.clone()).square() * 
                    (18.0 - 32.0*x.clone() + 12.0*x.clone().square() + 48.0*y.clone() - 
                        36.0*x.clone()*y.clone() + 27.0*y.clone().square()))
            }
            let x = Variable::new(1.0);
            let y = Variable::new(1.0);
            let z = goldstein(x.clone(), y.clone());
            z.backward();
            assert_eq!(x.get_grad_data(), Some(-5376.0));
            assert_eq!(y.get_grad_data(), Some(8064.0));
        }

        #[test]
        fn test_2nd_defferentiation() {
            // TODO: fix error in this test

            // unsafe { config_retain_off(); }
            fn f(x: Variable) -> Variable {
                x.clone().square().square() - 2.0 * x.clone().square()
            }
            let x = Variable::new(2.0);
            x.set_name("x");
            let y = f(x.clone());
            y.set_name("y");
            y.backward();
            assert_eq!(x.get_grad_data(), Some(24.0));

            let gx = x.get_grad().unwrap();
            gx.set_name("gx");
            x.clear_grad();
            gx.backward();
            assert_eq!(x.get_grad_data(), Some(44.0));
        }

        #[test]
        fn numerical_diff_test() {
            let dx = numerical_diff(&|x: f64| x*x , 2.0);
            assert!((dx - 4.0).abs() < 1e-6);
        }

    }


}


mod base64 {
    pub(super) fn to_f64(data: &[u8]) -> Vec<f64> {

        const BASE64_MAP: &[u8] =
            b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

        let mut stream = vec![];
        let mut cursor = 0;

        while cursor + 4 <= data.len() {
            let mut buffer = 0u32;

            for i in 0..4 {
                let c = data[cursor + i];
                let shift = 6 * (3 - i);

                for (i, &d) in BASE64_MAP.iter().enumerate() {
                    if c == d {
                        buffer |= (i as u32) << shift;
                    }
                }
            }

            for i in 0..3 {
                let shift = 8 * (2 - i);
                let value = (buffer >> shift) as u8;
                stream.push(value);
            }

            cursor += 4;
        }

        let mut result = vec![];
        cursor = 0;

        while cursor + 8 <= stream.len() {
            let p = stream.as_ptr() as *const f64;
            let x = unsafe { *p.offset(cursor as isize / 8) };
            result.push(x);
            cursor += 8;
        }

        result
    }

    pub(super) fn to_base64(data: &Vec<f64>) -> String {

        const BASE64_MAP: [char; 64] = [
            'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H',
            'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P',
            'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X',
            'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f',
            'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n',
            'o', 'p', 'q', 'r', 's', 't', 'u', 'v',
            'w', 'x', 'y', 'z', '0', '1', '2', '3',
            '4', '5', '6', '7', '8', '9', '+', '/',
        ];
        const PADDING: char = '=';

        let mut result = String::new();
        let mut buffer = 0u8;
        let mut shift = 0;

        for i in 0..data.len() {
            let byte_array = data[i].to_le_bytes();
            for j in 0..8 {
                shift = shift % 6 + 2;
                let c = 0b00111111 & (buffer | (byte_array[j] >> shift)) as u8; 
                for (i, d) in BASE64_MAP.iter().enumerate() {
                    if i as u8 == c {
                        result.push(*d);
                    }
                }
                
                buffer = 0b00111111 & (byte_array[j] << (6 - shift));
                // when buffer has 6bit,  
                if shift == 6 {
                    for (i, d) in BASE64_MAP.iter().enumerate() {
                        if i as u8 == buffer {
                            result.push(*d);
                        }
                    } 
                    buffer = 0;
                }
                
            }
        }

        // add residual buffer
        if shift > 0 {
            for (i, d) in BASE64_MAP.iter().enumerate() {
                if i as u8 == buffer {
                    result.push(*d);
                }
            }
        }

        while result.len() % 4 > 0 {
            result.push(PADDING);
        } 

        result
    }
    
    #[cfg(test)]
    mod test {
        use crate::base64::{to_base64, to_f64};

        #[test]
        fn test_base64() {
            assert_eq!(to_f64(b"AAAAAN7lsD8AAADATM/Yvw=="), vec![0.06600749492645264, -0.38765257596969604]);
            assert_eq!(to_f64(b"AAAAAAAALkA="), vec![15.0f64]);
            assert_eq!(to_base64(&vec![15.0f64]), String::from("AAAAAAAALkA="));
            assert_eq!(to_base64(&vec![0.06600749492645264, -0.38765257596969604]), String::from("AAAAAN7lsD8AAADATM/Yvw=="));
        }

    }

}
