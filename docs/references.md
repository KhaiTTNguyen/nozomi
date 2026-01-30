## Numerically Stable Quadratic Equation Solving
Collision detection was obtained from solving the quadratic equation for sphere-ray intersections. 

### Problem: 
Collision detection requires solving: 
```bash
ax² + bx + c = 0 where:
a = jx² + jy² + jz² (squared jump magnitude)
b = 2(jx·dx + jy·dy + jz·dz) (dot product terms)
c = dx² + dy² + dz² - r² (distance to sphere surface)

### Naive Implementation Issues
// UNSTABLE - suffers from catastrophic cancellation
float discriminant = sqrt(b*b - 4*a*c);
float x1 = (-b + discriminant) / (2*a);
float x2 = (-b - discriminant) / (2*a);
```
But, standard formula `(-b ± √(b²-4ac))/2a` suffers from catastrophic cancellation when `b²≫4ac`

### Approach: 
Muller's method was used to avoid catastrophic cancellation:
* The stable version always adds terms of the same sign in the numerator
* Uses Vieta's formulas (x₁x₂ = c/a) to get the second root

```bash
float sqterm = sqrt(b*b - 4*a*c);
float q = -0.5 * (b + copysign(sqterm, b));
float x1 = q / a;
float x2 = c / q;
```
## Tolerance Handling Around Sphere Boundaries
### Problem:
If molecule's step ends exactly on boundary, adding a tolerance prevents particles from getting "stuck" exactly on boundaries.

### Approach: 
Create a "transition zone" of width `2 × tol` around each sphere
* Inner tolerance: `r-tol`
* Outer tolerance: `r+tol`

#### Tolerance Selection Rationale
A 32-bit single-precision float has ~7 decimal digits
Our tolerance `tol = 1e-4` provides safety margin while avoiding excessive buffer zones.

This tolerance buffer around each sphere was implemented as:
```bash
const float tol = 1e-4;  // Global tolerance

// Inside sphere: collision with (r - tol)
float c = dx*dx + dy*dy + dz*dz - (r-tol)*(r-tol);

// Outside sphere: collision with (r + tol) 
float c = dx*dx + dy*dy + dz*dz - (r+tol)*(r+tol);
```

Order of structure ids.
and references


Structuring project:
https://cookiecutter-data-science.drivendata.org/

To create `requirements.txt` file or update dependencies in `requirements.txt`:

`pip freeze > requirements.txt` 

For Ubuntu 22, use numpy-1.22.4 & scipy-1.7.2

Check GPU util by

`nvitop` to output GPU snapshot every 0.1s 