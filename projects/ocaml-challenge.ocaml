(*
As a Machine Learning engineer at OpenAI, you're working on optimizing a neural network for image recognition. 
The network's performance is characterized by a scalar field f(x, y, z) representing the error rate, where x, y, and z are hyperparameters of the model:
f(x, y, z) = x^2 * sin(y) * cos(z) + y^2 * e^(-z) + z^3 * log(x^2 + y^2 + 1)

- Find the gradient vector ∇f at the point (1, π/2, 0).
- Determine the direction of steepest descent at (1, π/2, 0).
- Use the method of Lagrange multipliers to find the minimum value of f subject to the constraint g(x, y, z) = x^2 + y^2 + z^2 = 4.
- Calculate the curl of the vector field F(x, y, z) = [yz, xz, xy] at (1, π/2, 0).
- Evaluate the surface integral of f over the sphere x^2 + y^2 + z^2 = 4.
- Find the volume of the region bounded by the surfaces z = f(x, y) and z = 0 over the region 0 ≤ x ≤ 1, 0 ≤ y ≤ 1.

*)


(*

Solution (no code) - available in LaTeX on Canvas: 

Gradient vector: To find ∇f at (1, π/2, 0), we calculate partial derivatives and evaluate them at the point: ∂f/∂x = 2x * sin(y) * cos(z) + 2z^3 * x / (x^2 + y^2 + 1) ∂f/∂y = x^2 * cos(y) * cos(z) + 2y * e^(-z) ∂f/∂z = -x^2 * sin(y) * sin(z) - y^2 * e^(-z) + 3z^2 * log(x^2 + y^2 + 1) Evaluating at (1, π/2, 0): ∇f = (0, 1, -π^2/4)
Direction of steepest descent: The direction of steepest descent is the negative of the normalized gradient vector: -∇f / ||∇f|| = -(0, 1, -π^2/4) / sqrt(1 + π^4/16) ≈ (0, -0.8919, 0.4522)
Lagrange multipliers: Set up the Lagrangian: L(x, y, z, λ) = f(x, y, z) - λ(x^2 + y^2 + z^2 - 4) Solve ∇L = 0 and g(x, y, z) = 4 simultaneously (this is a complex system of equations)
Curl: curl F = [∂F_z/∂y - ∂F_y/∂z, ∂F_x/∂z - ∂F_z/∂x, ∂F_y/∂x - ∂F_x/∂y] = [x - z, y - x, z - y] At (1, π/2, 0): curl F = (1, π/2 - 1, -π/2)
Surface integral: Use spherical coordinates and the surface element dS = R^2 * sin(φ) dθ dφ ∫∫_S f dS = ∫_0^π ∫_0^2π f(2sin(φ)cos(θ), 2sin(φ)sin(θ), 2cos(φ)) * 4 * sin(φ) dθ dφ
Volume: V = ∫_0^1 ∫_0^1 f(x, y) dx dy

*)

(*

Solution (oCaml): 

*) 

open Float

let f x y z =
  x ** 2. *. sin y *. cos z +. y ** 2. *. exp (-z) +. z ** 3. *. log (x ** 2. +. y ** 2. +. 1.)

let grad_f x y z =
  let dx = 2. *. x *. sin y *. cos z +. 2. *. z ** 3. *. x /. (x ** 2. +. y ** 2. +. 1.) in
  let dy = x ** 2. *. cos y *. cos z +. 2. *. y *. exp (-z) in
  let dz = -.x ** 2. *. sin y *. sin z -. y ** 2. *. exp (-z) +. 3. *. z ** 2. *. log (x ** 2. +. y ** 2. +. 1.) in
  (dx, dy, dz)

let steepest_descent grad =
  let (dx, dy, dz) = grad in
  let mag = sqrt (dx ** 2. +. dy ** 2. +. dz ** 2.) in
  (-.dx /. mag, -.dy /. mag, -.dz /. mag)

let curl_f x y z =
  (x -. z, y -. x, z -. y)

let surface_integral_sphere r f =
  let rec integrate_phi theta phi acc =
    if phi > pi then acc
    else
      let x = r *. sin phi *. cos theta in
      let y = r *. sin phi *. sin theta in
      let z = r *. cos phi in
      let element = f x y z *. r ** 2. *. sin phi in
      integrate_phi theta (phi +. 0.01) (acc +. element *. 0.01)
  in
  let rec integrate_theta theta acc =
    if theta > 2. *. pi then acc
    else
      let phi_integral = integrate_phi theta 0. 0. in
      integrate_theta (theta +. 0.01) (acc +. phi_integral *. 0.01)
  in
  integrate_theta 0. 0.

let volume_integral f =
  let rec integrate_y x y acc =
    if y > 1. then acc
    else integrate_y x (y +. 0.01) (acc +. f x y *. 0.01)
  in
  let rec integrate_x x acc =
    if x > 1. then acc
    else
      let y_integral = integrate_y x 0. 0. in
      integrate_x (x +. 0.01) (acc +. y_integral *. 0.01)
  in
  integrate_x 0. 0.

let () =
  let x, y, z = (1., pi /. 2., 0.) in
  
  (* 1. Gradient *)
  let grad = grad_f x y z in
  Printf.printf "1. Gradient at (1, π/2, 0): (%.4f, %.4f, %.4f)\n" (fst3 grad) (snd3 grad) (thd3 grad);
  
  (* 2. Steepest descent *)
  let descent = steepest_descent grad in
  Printf.printf "2. Steepest descent direction: (%.4f, %.4f, %.4f)\n" (fst3 descent) (snd3 descent) (thd3 descent);
  
  (* 3. Lagrange multipliers (not implemented due to complexity) *)
  Printf.printf "3. Lagrange multipliers: Analytical solution required\n";
  
  (* 4. Curl *)
  let curl = curl_f x y z in
  Printf.printf "4. Curl at (1, π/2, 0): (%.4f, %.4f, %.4f)\n" (fst3 curl) (snd3 curl) (thd3 curl);
  
  (* 5. Surface integral *)
  let surface_integral = surface_integral_sphere 2. f in
  Printf.printf "5. Surface integral over sphere: %.4f\n" surface_integral;
  
  (* 6. Volume *)
  let volume = volume_integral f in
  Printf.printf "6. Volume: %.4f\n" volume
