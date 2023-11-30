import React from "react";

const WelcomeSection: React.FC = () => {
  return (
    <section className="about-section mt-5" id="Welcome_nav">
      <div className="container">
        <div className="row justify-content-center">
          <div className="col-lg-6 text-center text-uppercase font-bold mb-5">
            <h2 className="text-black mb-2">Welcome To Tourismo</h2>
            <p>
              Lorem ipsum dolor sit amet consectetur adipisicing elit. Quasi a
              debitis et dolore ipsam minus voluptatum dolorum corporis
              doloremque ea.
            </p>
          </div>
        </div>

        {/* Culture */}
        <div className="flex items-center mb-5">
          <div className="w-full md:w-1/2 md:order-2">
            <a href="#">
              <img
                src="https://i.postimg.cc/P5n01vCY/about-1.jpg"
                alt=""
                className="w-full h-full object-cover"
              />
            </a>
          </div>
          <div className="w-full md:w-1/2 md:mr-auto text-center md:text-left">
            <h2 className="text-black font-bold">Culture</h2>
            <p className="mb-4">
              Lorem ipsum dolor sit amet consectetur adipisicing elit. Cum
              minima amet, assumenda aliquid temporibus molestiae.
            </p>
            <a
              href="#"
              className="bg-blue-900 text-white mx-2 px-4 py-1 rounded-md"
            >
              Read More
            </a>
          </div>
        </div>

        {/* History */}
        <div className="flex items-center mb-5">
          <div className="w-full md:w-1/2 md:order-1">
            {/* Change md:order-2 to md:order-1 */}
            <a href="#">
              <img
                src="https://i.postimg.cc/1XjnLZq1/about-2.jpg"
                alt=""
                className="w-full h-full object-cover"
              />
            </a>
          </div>
          <div className="w-full md:w-1/2 md:mr-auto text-center md:text-left">
            <h2 className="text-black font-bold">History</h2>
            <p className="mb-4">
              Lorem ipsum dolor sit amet consectetur adipisicing elit. Cum
              minima amet, assumenda aliquid temporibus molestiae.
            </p>
            <a
              href="#"
              className="bg-blue-900 text-white mx-2 px-4 py-1 rounded-md"
            >
              Read More
            </a>
          </div>
        </div>

        {/* Nature */}
        <div className="flex items-center mb-5">
          <div className="w-full md:w-1/2 md:order-2">
            {/* Change md:order-2 to md:order-1 */}
            <a href="#">
              <img
                src="https://i.postimg.cc/1X9r9PXg/about-3.jpg"
                alt=""
                className="w-full h-full object-cover"
              />
            </a>
          </div>
          <div className="w-full md:w-1/2 md:mr-auto text-center md:text-left">
            <h2 className="text-black font-bold">Nature</h2>
            <p className="mb-4">
              Lorem ipsum dolor sit amet consectetur adipisicing elit. Cum
              minima amet, assumenda aliquid temporibus molestiae.
            </p>
            <a
              href="#"
              className="bg-blue-900 text-white mx-2 px-4 py-1 rounded-md"
            >
              Read More
            </a>
          </div>
        </div>
      </div>
    </section>
  );
};

export default WelcomeSection;
