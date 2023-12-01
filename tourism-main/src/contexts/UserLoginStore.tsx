import axios from "axios";
import { createContext, ReactNode, useContext, useState } from "react";
import { useNavigate } from "react-router-dom";

// Define the types
interface User {
  userId: number;
  username: string;
  userEmail: string;
  password?: string;
}

interface LoginContextProps {
  currentUser: User | null;
  loginError: string;
  userLoginStatus: boolean;
  loginUser: (userCredentialsObj: any) => void;
  setUserLoginStatus: React.Dispatch<React.SetStateAction<boolean>>; // New function
}

const loginContext = createContext<LoginContextProps | undefined>(undefined);

// Create the hook for using the context
export const useLoginContext = () => {
  const context = useContext(loginContext);
  if (!context) {
    throw new Error("useLoginContext must be used within a LoginProvider");
  }
  return context;
};

function UserLoginStore({ children }: { children: ReactNode }) {
  const [currentUser, setCurrentUser] = useState<User | null>(null);
  const [loginError, setLoginError] = useState("");
  const [userLoginStatus, setUserLoginStatus] = useState(false);
  const navigate = useNavigate();

  const loginUser = (userCredentialsObj: any) => {
    // Set loading state if needed
    // setLoading(true);

    axios
      .post("http://localhost:3002/user/login", userCredentialsObj)
      .then((response) => {
        if (response.data.data.success === true) {
          setCurrentUser({ ...response.data.data.data });
          setUserLoginStatus(true);
          setLoginError("");
          navigate("/");
        } else {
          setLoginError(response.data.data.message);
          setUserLoginStatus(false);
        }
      })
      .catch((err) => {
        setLoginError("An error occurred during login.");
      });
  };

  // Add setUserLoginStatus to the context value
  const contextValue: LoginContextProps = {
    currentUser,
    loginError,
    userLoginStatus,
    loginUser,
    setUserLoginStatus,
  };

  return (
    <loginContext.Provider value={contextValue}>
      {children}
    </loginContext.Provider>
  );
}

export default UserLoginStore;
