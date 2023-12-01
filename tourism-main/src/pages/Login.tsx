import React, { useContext, useState } from "react";
import { useNavigate } from "react-router-dom";
import { userLogin } from "../APISs/usersapi";
import { loginContext } from "../contexts/loginContext";
import { useLoginContext } from "../contexts/UserLoginStore";

const Login: React.FC = () => {
  const navigate = useNavigate();
  const [formData, setFormData] = useState({
    email: "",
    password: "",
  });
  const { currentUser, loginError, userLoginStatus, loginUser } =
    useLoginContext();

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const { name, value } = e.target;
    setFormData((prevData) => ({
      ...prevData,
      [name]: value,
    }));
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    const payload = {
      userEmail: formData.email,
      password: formData.password,
    };
    loginUser(payload);
  };

  return (
    <div className="min-h-screen flex items-center justify-center">
      <form className="max-w-md w-full p-6 bg-white rounded-md shadow-md">
        <h2 className="text-2xl font-semibold mb-6">Login</h2>
        {loginError.length !== 0 && (
          <p className="text-red-600 text-center">*{loginError}</p>
        )}
        <div className="mb-4">
          <label
            htmlFor="email"
            className="block text-gray-600 text-sm font-medium mb-2"
          >
            Email
          </label>
          <input
            type="email"
            id="email"
            name="email"
            value={formData.email}
            onChange={handleChange}
            className="w-full border border-gray-300 rounded-md p-2"
          />
        </div>

        <div className="mb-4">
          <label
            htmlFor="password"
            className="block text-gray-600 text-sm font-medium mb-2"
          >
            Password
          </label>
          <input
            type="password"
            id="password"
            name="password"
            value={formData.password}
            onChange={handleChange}
            className="w-full border border-gray-300 rounded-md p-2"
          />
        </div>

        <button
          type="submit"
          onClick={handleSubmit}
          className="bg-blue-500 text-white px-4 py-2 rounded-md hover:bg-blue-600 transition duration-300"
        >
          Login
        </button>
        <div className="text-[14px]">
          <div className="flex justify-center">
            <div>New user?</div>
            <div
              className="text-blue-900 underline cursor-pointer hover:text-[#8785ee]"
              onClick={() => navigate(`/signup`)}
            >
              {" "}
              SignUp here
            </div>
          </div>
        </div>
      </form>
    </div>
  );
};

export default Login;
