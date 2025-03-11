# Submittion:

## Repos:
- [Our mp1 Repo](https://github.com/c-mawn/fun-robo-mp1)
- [Our Forked Hiwonder Repo (With our code)](https://github.com/Zaraius/hiwonder-armpi-pro)
- [Arm Kinematics Visualizer Repo](https://github.com/OlinCollege-FunRobo/arm-kinematics-module)
- [Hiwonder Robot Repo](https://github.com/OlinCollege-FunRobo/hiwonder-armpi-pro)


## Robot Arm Diagram -- with frames
[diagram](https://www.dropbox.com/scl/fi/qj3w9asxu9amepujsrdbi/funrobo_mp1_diagram.png?rlkey=yy91buuaqyxy2pn6q56p197ck&st=6jyxbsdd&dl=0)
![diagram](/assets/funrobo_mp1_diagram.png)
## Derivation of DH Table and Jacobian Matricies
[math](https://www.dropbox.com/scl/fi/c9phjtv2rnwxzsmpd83a9/funrobo_mp1_math.png?rlkey=ljtr6bcz4tgt2mo99xcepi7fe&st=601i9k6g&dl=0)
![math](/assets/funrobo_mp1_math.png)
## Video Demos

### FPK Visualizer Video
[FPK Visualizer Video](https://youtu.be/C8_INBxMgc4)



### FVK Visualizer Video
[FVK Visualizer Video](https://youtu.be/aM-NqXCQ0vk)



### Robot Arm Moving in 3 Axes Video
[Robot Arm Demo (3-axes) Video](https://youtu.be/yc-X9xhSfXw)



### Robot Arm Picking up Block Video
[Robot Arm Demo (picking up blocks) Video](https://www.youtube.com/watch?v=wYRfEQr1miI)



(Sorry that last video is sideways)

## Code

All implemented function can be found in the `arm_models.py` file under the `FiveDOFRobot` class, As well as below:


```python
class FiveDOFRobot:
    """
    A class to represent a 5-DOF robotic arm with kinematics calculations, including
    forward kinematics, inverse kinematics, velocity kinematics, and Jacobian computation.

    Attributes:
        l1, l2, l3, l4, l5: Link lengths of the robotic arm.
        theta: List of joint angles in radians.
        theta_limits: Joint limits for each joint.
        ee: End-effector object for storing the position and orientation of the end-effector.
        num_dof: Number of degrees of freedom (5 in this case).
        points: List storing the positions of the robot joints.
        DH: Denavit-Hartenberg parameters for each joint.
        T: Transformation matrices for each joint.
    """

    def __init__(self):
        """Initialize the robot parameters and joint limits."""
        # Link lengths
        self.l1, self.l2, self.l3, self.l4, self.l5 = 0.30, 0.15, 0.18, 0.15, 0.12

        # Joint angles (initialized to zero)
        self.theta = [0, 0, 0, 0, 0]

        # Joint limits (in radians)
        self.theta_limits = [
            [-np.pi, np.pi],
            [-np.pi / 3, np.pi],
            [-np.pi + np.pi / 12, np.pi - np.pi / 4],
            [-np.pi + np.pi / 12, np.pi - np.pi / 12],
            [-np.pi, np.pi],
        ]

        # End-effector object
        self.ee = EndEffector()

        # Robot's points
        self.num_dof = 5
        self.points = [None] * (self.num_dof + 1)

        # Denavit-Hartenberg parameters and transformation matrices

        self.H05 = np.matrix(
            [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
        )  # Denavit-Hartenberg parameters (theta, d, a, alpha)
        # Transformation matrices

        self.H_01 = np.empty((4, 4))
        self.H_12 = np.empty((4, 4))
        self.H_23 = np.empty((4, 4))
        self.H_34 = np.empty((4, 4))
        self.H_45 = np.empty((4, 4))

        self.T = np.zeros((self.num_dof, 4, 4))

    def calc_forward_kinematics(self, theta: list, radians=False):
        """
        Calculate forward kinematics based on the provided joint angles.

        Args:
            theta: List of joint angles (in degrees or radians).
            radians: Boolean flag to indicate if input angles are in radians.
        """
        if not radians:
            theta = [np.deg2rad(angle) for angle in theta]

        self.H_01 = dh_to_matrix([theta[0], self.l1, 0, -PI / 2])
        self.H_12 = dh_to_matrix([theta[1] - PI / 2, 0, self.l2, PI])
        self.H_23 = dh_to_matrix([theta[2], 0, self.l3, PI])
        self.H_34 = dh_to_matrix([theta[3] + PI / 2, 0, 0, PI / 2])
        self.H_45 = dh_to_matrix([theta[4], self.l4 + self.l5, 0, 0])

        self.H05 = self.H_01 @ self.H_12 @ self.H_23 @ self.H_34 @ self.H_45
        self.T = [self.H_01, self.H_12, self.H_23, self.H_34, self.H_45]

        # Calculate robot points (positions of joints)
        self.calc_robot_points()

    def jacobian_v(self):
        """
        Calculates the Jacobian Matrix at the given thetas
        """

        J = np.empty((3, 5))

        t1 = self.H05[0:3, 3]
        r1 = np.array([0, 0, 1])
        j1 = np.cross(r1, t1)
        J[0:3, 0] = j1

        t2 = self.H05[0:3, 3] - self.H_01[0:3, 3]
        r2 = self.H_01[0:3, 2]
        j2 = np.cross(r2, t2)
        J[0:3, 1] = j2

        t3 = self.H05[0:3, 3] - (self.H_01 @ self.H_12)[0:3, 3]
        r3 = (self.H_01 @ self.H_12)[0:3, 2]
        j3 = np.cross(r3, t3)
        J[0:3, 2] = j3

        t4 = self.H05[0:3, 3] - (self.H_01 @ self.H_12 @ self.H_23)[0:3, 3]
        r4 = (self.H_01 @ self.H_12 @ self.H_23)[0:3, 2]
        j4 = np.cross(r4, t4)
        J[0:3, 3] = j4

        t5 = self.H05[0:3, 3] - (self.H_01 @ self.H_12 @ self.H_23 @ self.H_34)[0:3, 3]
        r5 = (self.H_01 @ self.H_12 @ self.H_23 @ self.H_34)[0:3, 2]
        j5 = np.cross(r5, t5)
        J[0:3, 4] = j5

        return J

    def inverse_jacobian(self):
        """
        Creates the inverse jacobian matrix based on the jacobian.

        Returns:
            the pseudo inverse of the jacobian matrix
        """
        J = self.jacobian_v()

        # Calculate pinv of the jacobian
        lambda_constant = 0.01
        J_inv = np.transpose(J) @ np.linalg.inv(
            ((J @ np.transpose(J)) + lambda_constant**2 * np.identity(3))
        )

        return J_inv

    def calc_velocity_kinematics(self, vel: list):
        """
        Calculate the joint velocities required to achieve the given end-effector velocity.

        Args:
            vel: Desired end-effector velocity (3x1 vector).
        """
        ########################################
        # at every time step, inverse inverse jacboan * cartesian
        time_step = 0.01
        q_dot = self.inverse_jacobian() @ np.array(vel)
        self.theta = self.theta + time_step * np.array(q_dot)

        ########################################

        # Recompute robot points based on updated joint angles
        self.calc_forward_kinematics(self.theta, radians=True)

    def calc_robot_points(self):
        """Calculates the main arm points using the current joint angles"""

        # Initialize points[0] to the base (origin)
        self.points[0] = np.array([0, 0, 0, 1])

        # Precompute cumulative transformations to avoid redundant calculations
        T_cumulative = [np.eye(4)]
        for i in range(self.num_dof):
            T_cumulative.append(T_cumulative[-1] @ self.T[i])

        # Calculate the robot points by applying the cumulative transformations
        for i in range(1, 6):
            self.points[i] = T_cumulative[i] @ self.points[0]

        # Calculate EE position and rotation
        self.EE_axes = T_cumulative[-1] @ np.array(
            [0.075, 0.075, 0.075, 1]
        )  # End-effector axes
        self.T_ee = T_cumulative[-1]  # Final transformation matrix for EE

        # print(self.points)

        # Set the end effector (EE) position
        self.ee.x, self.ee.y, self.ee.z = self.points[-1][:3]

        # Extract and assign the RPY (roll, pitch, yaw) from the rotation matrix
        rpy = rotm_to_euler(self.T_ee[:3, :3])
        self.ee.rotx, self.ee.roty, self.ee.rotz = rpy[2], rpy[1], rpy[0]

        # Calculate the EE axes in space (in the base frame)
        self.EE = [self.ee.x, self.ee.y, self.ee.z]
        self.EE_axes = np.array(
            [self.T_ee[:3, i] * 0.075 + self.points[-1][:3] for i in range(3)]
        )
```

All code related to the physical implementation of the project is the same as above, just pushed into the repo for the physical robot. However, we did write a few lines in that repo, which is written below

This code below can be found in our [Hiwonder Repo](https://github.com/Zaraius/hiwonder-armpi-pro), in the `hiwonder.py` file, class `HiwonderRobot` under the function `set_arm_velocity`:

```python
bot = ut.FiveDOFRobot()
bot.theta = self.joint_values[:-1]
thetalist_dot = (bot.inverse_jacobian()) @ np.array(vel)
```

## Individual Reflections

### Zaraius

#### What did you learn from this? What did you not know before this assignment?
From this assignment I learnt a lot about translating theoretical math into code which controls a robot. Before this I had a very basic understanding of numpy but after a lot of time spent debugging I have a more concrete understanding of the library.
#### What was the most difficult aspect of the assignment?
The most difficult aspect of the assignment for me was debugging various aspects of the code while trying to get the simulator to run. For example our DH matrix was of the data type np.matrix instead of np.array which took hours to find. Another bug we faced was the robot was counting angles in degrees but the dh_to_matrix wanted them in radians. Without this conversion the robot would move sparatically and we tried changing many other aspects of our code until we tried this.   
#### What was the easiest or most straightforward aspect of the assignment?
The easiest aspect of the assignment was converting the simulator code into working robot code. I think it was easy because other people had already figured out a lot of the small issues with the robot so I didn't have to spend too much time on it. 
#### How long did this assignment take? What took the most time (PC setup? Coding in Python? Exploring the questions?)?
The assignment took about 15 hours to complete. Most of the time was spent on coding and debugging python code. We also spent a considerable amount of time deriving our DH matrix and the jacobian.
#### What did you learn about arm forward kinematics that we didn't explicitly cover in class?
During this assignment I learnt a geometric way to calculate the jacobian which we didn't cover in class. We learnt this approach by first asking Kene who showed it to us and then trying it on our own and then Dominic helped us solidify it.
#### What more would you like to learn about arm forward kinematics?
I would like to learn forward kinematics for differnt type of robot arms. For example I know that trying to controlling a delta arm would be different than a SCARA or a 5-dof. I would like to learn how to control parallel mechanism with multiple arms.

### Charlie

#### What did you learn from this? What did you not know before this assignment?
I personally learned a lot about the math behind robot arm movement. I had no experience with H matricies, DH Tables, or Jacobians before this project. I also learned quite a bit about using ssh to connect with a raspi. 
#### What was the most difficult aspect of the assignment?
For me, the most difficult part of this assignment was understanding the mathematical portion of the project, as well as debugging the math implementation in code. 
#### What was the easiest or most straightforward aspect of the assignment?
I think the easiest part of the assignment was writing the singularity avoidance code. I was intimidated at first but it was really simple to just change one line of code to be a slightly more complicated equation for calculating the inverse jacobian. 
#### How long did this assignment take? What took the most time (PC setup? Coding in Python? Exploring the questions?)?
I think we spent around 15-20 hours on the project. The biggest time sink was probably the implementation and debugging of the visualizer. 
#### What did you learn about arm forward kinematics that we didn't explicitly cover in class?
I feel like I didn't explicitly learn anything new, but I am now able to understand the majority of the content. When we first learned it, I was absolutely lost and had no idea what was going on, but now, I generally understand all the pieces that went into making this project work. 
#### What more would you like to learn about arm forward kinematics?
I think the biggest thing I want to learn is how people actually use this in real life? Is it more often that people just use inverse kinematics? Or is fk just as commonly used as ik. 

### Alex

#### What did you learn from this? What did you not know before this assignment?
From this project, I got a much better understanding of robot kinematics, as well as how to implement them in MATLAB/Python code. Before this assignment, I had a vague understanding of FPK/FVK from the lectures, but implementing them helped me understand them conceptually and visualize what was actually happening.

#### What was the most difficult aspect of the assignment?
The most difficult aspect of this assignment for me was creating the DH tables, as it took a while for me to understand what each value was supposed to physically represent.

#### What was the easiest or most straightforward aspect of the assignment?
The most straightforward part of the assignment was the RRMC, as even though we had issues with our Jacobian matrix, the calculations and step commands around it were pretty simple.

#### How long did this assignment take? What took the most time (PC setup? Coding in Python? Exploring the questions?)?
This project took around 20 hours. I felt like we spent the most time deriving the DH table in the beginning.

#### What did you learn about arm forward kinematics that we didn't explicitly cover in class?
While I didn't specifically learn anything new, I had a much better understanding of concepts such as RRMC through my own research in this process rather than what we covered in class.

#### What more would you like to learn about arm forward kinematics?
I would like to solidify my core understanding of the topic before moving on, but I would like to go more into singularities and how to avoid them.