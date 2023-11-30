import React from "react";
import { NavLink, useNavigate } from "react-router-dom";
import { useLoginContext } from "../contexts/UserLoginStore";

const Navbar: React.FC = () => {
  const {
    currentUser,
    loginError,
    userLoginStatus,
    loginUser,
    setUserLoginStatus,
  } = useLoginContext();
  return (
    <div className="bg-[#077cb3] p-4">
      <div className="container mx-auto flex justify-between items-center">
        <div className="text-[22px]">Tourismo</div>
        <form>
          <input
            className="p-2 rounded-md mr-5"
            type="search"
            placeholder="Search"
            aria-label="Search"
          />
          <button
            className="border border-green-800 text-green-800 p-2 rounded-lg"
            type="submit"
          >
            Search
          </button>
        </form>
        <div className="flex space-x-4">
          <NavLink
            className="text-white cursor-pointer hover:text-yellow-400"
            to={`/`}
          >
            Home
          </NavLink>
          <NavLink
            className="text-white cursor-pointer hover:text-yellow-400"
            to={`/about`}
          >
            About
          </NavLink>
          <NavLink
            className="text-white cursor-pointer hover:text-yellow-400"
            to={`/popularplaces`}
          >
            Popular Places
          </NavLink>
          {/* <NavLink
            className="text-white cursor-pointer hover:text-yellow-400"
            to={`/servicessection`}
          >
            Services
          </NavLink> */}
          <NavLink
            className="text-white cursor-pointer hover:text-yellow-400"
            to={`/historicalplaces`}
          >
            Historical Places
          </NavLink>
          {/* <NavLink
            className="text-white cursor-pointer hover:text-yellow-400"
            to={`/teamsection`}
          >
            Team
          </NavLink> */}
          <div className="text-white cursor-pointer hover:text-yellow-400">
            Contact
          </div>
          {userLoginStatus === false ? (
            <NavLink
              className="text-white cursor-pointer hover:text-yellow-400"
              to={`/login`}
            >
              Login/SignUp
            </NavLink>
          ) : (
            <div
              className="text-white cursor-pointer hover:text-yellow-400"
              onClick={() => setUserLoginStatus(false)}
            >
              Logout
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default Navbar;
