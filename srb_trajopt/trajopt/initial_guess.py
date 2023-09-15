from dataclasses import dataclass
import numpy as np

class SRBTrajoptInitialGuess:
    """
    Container for storing the initial guess for the SRB trajectory. 
    The contact mode sequence defined by the left and right foot contacts 
    are required for non-contact implicit optimization.

    While it is possible for a "foot" of the SRB model to contact the ground with less 
    than the total number of contact points per foot, left and right foot contact sequences 
    will assume that all contact points of the foot are in contact with the ground if a 1 is given
    in the contact sequence, not in contact if a 0 is given instead.

    Optional arguments for the SRB states will use default values if not provided.
    Args:
        left_foot_contacts: binary vector of left foot contact status
        right_foot_contacts: binary vector of right foot contact status
        N: number of knot points
        T: total time
        com_orientation_guess: initial guess for the CoM orientation of shape (4,N)
        com_pos_guess: initial guess for the CoM position of shape (3,N)
        com_angular_vel_guess: initial guess for the CoM angular velocity of shape (3,N)
        com_vel_guess: initial guess for the CoM velocity of shape (3,N)
        p_W_L_guess: initial guess for the left foot position of shape (3,N)
    """
    left_foot_contacts: np.ndarray
    right_foot_contacts: np.ndarray
    N: int
    T: float
    com_orientation_guess = None
    com_position_guess = None
    com_angular_velocity_guess = None
    com_velocity_guess = None
    p_W_LF_guess = None
    p_W_RF_guess = None

    def __init__(self, left_foot_contacts, right_foot_contacts, N, T):
        # initialize required fields
        self.left_foot_contacts = left_foot_contacts
        self.right_foot_contacts = right_foot_contacts
        self.N = N
        self.T = T
        # Set default values for optional fields
        self._default_quaternion = np.array([[1., 0., 0., 0.]])
        self._default_position = np.array([[0., 0., 1.]])
        self._default_angular_velocity = np.array([[0., 0., 0.]])
        self._default_velocity = np.array([[0., 0., 0.]])
        self._default_p_W_LF = np.array([[0., 0.1, 0.]])
        self._default_p_W_RF = np.array([[0., -0.1, 0.]])

        self.com_orientation_guess = np.repeat(self._default_quaternion.T, self.N, axis=1)
        self.com_position_guess = np.repeat(self._default_position.T, self.N, axis=1)
        self.com_angular_velocity_guess = np.repeat(self._default_angular_velocity.T, self.N, axis=1)
        self.com_velocity_guess = np.repeat(self._default_velocity.T, self.N, axis=1)
        self.p_W_LF_guess = np.repeat(self._default_p_W_LF.T, self.N, axis=1)
        self.p_W_RF_guess = np.repeat(self._default_p_W_RF.T, self.N, axis=1)

    def __post_init__(self):
        ""
        input_arg_lengths = []

        if not np.all(np.isin(self.left_foot_contacts, [0, 1])) or \
            not np.all(np.isin(self.right_foot_contacts, [0, 1])):
            raise ValueError("left/right foot contact sequences must be a binary vector")
        
        assert self.left_foot_contacts.ndim == 1, "left_foot_contacts must be a 1D vector"
        assert self.right_foot_contacts.ndim == 1, "right_foot_contacts must be a 1D vector"
        input_arg_lengths.append(self.left_foot_contacts.shape[0])
        input_arg_lengths.append(self.right_foot_contacts.shape[0])

        if any(x != self.N for x in input_arg_lengths):
            raise ValueError("left/right foot contact sequences must be of length N")
    
    def _check_guess_data(self, data_target: np.ndarray, data_guess: np.ndarray):
        """
        Checks for invalid initial guess shapes
        """
        assert np.array_equal(data_target.shape, data_guess.shape), \
            f"Input guess with shape {data_guess.shape} must match default value of shape {data_target.shape}. \
             The default value second dimension is set by the value of knot points N on construction."
    
        # check for Nones or NaNs
        assert not np.any(np.isnan(data_guess)), "Initial guess cannot contain NaNs"
        assert not np.any(np.equal(data_guess, None)), "Initial guess cannot contain Nones"

        # all values must be double or float
        assert np.all(np.isin(data_guess.dtype, [np.float64, np.float32])), \
            "Initial guess must be of type double or float"
    
    def set_com_orientation_guess(self, com_orientation_guess: np.ndarray):
        self._check_guess_data(
            data_target=self.com_orientation_guess, 
            data_guess=com_orientation_guess) 

        # In addition, check that each quaternion guess entry is a unit quaternion
        for quat in com_orientation_guess.T:
            assert np.allclose(np.linalg.norm(quat), 1., atol=1e-4), \
            "com_orientation_guess must be a unit quaternion"

        self.com_orientation_guess = com_orientation_guess
    
    def set_com_position_guess(self, com_position_guess: np.ndarray):
        self._check_guess_data(
            data_target=self.com_position_guess, 
            data_guess=com_position_guess) 

        self.com_position_guess = com_position_guess
    
    def set_com_angular_velocity_guess(self, com_angular_velocity_guess: np.ndarray):
        self._check_guess_data(
            data_target=self.com_angular_velocity_guess, 
            data_guess=com_angular_velocity_guess) 

        self.com_angular_velocity_guess = com_angular_velocity_guess
    
    def set_com_velocity_guess(self, com_velocity_guess: np.ndarray):
        self._check_guess_data(
            data_target=self.com_velocity_guess, 
            data_guess=com_velocity_guess) 

        self.com_velocity_guess = com_velocity_guess
    
    def set_p_W_LF_guess(self, p_W_LF_guess: np.ndarray):
        self._check_guess_data(
            data_target=self.p_W_LF_guess, 
            data_guess=p_W_LF_guess) 

        self.p_W_LF_guess = p_W_LF_guess
    
    def set_p_W_RF_guess(self, p_W_RF_guess: np.ndarray):
        self._check_guess_data(
            data_target=self.p_W_RF_guess, 
            data_guess=p_W_RF_guess) 

        self.p_W_RF_guess = p_W_RF_guess