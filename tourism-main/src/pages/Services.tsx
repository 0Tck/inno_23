import React from 'react';

const Services: React.FC = () => {
  return (
    <section id="services" className="services section-bg">
      <div className="container">
        <div className="flex flex-wrap">
          <div className="col-lg-4">
            <div className="section-title">
              <h2>Services We Provide</h2>
              <p>
                Lorem ipsum dolor sit amet consectetur, adipisicing elit. Quisquam, rem ad id aspernatur dolores animi!
              </p>
            </div>
          </div>

          <div className="col-lg-8">
            <div className="flex flex-wrap">
              {/* Service 1 */}
              <div className="col-md-6 d-flex flex-col items-stretch">
                <div className="icon-circle purple hover:border-purple-600 hover:shadow-lg">
                  <div className="icon">
                    <i className="fas fa-drumstick-bite text-white text-3xl"></i>
                  </div>
                  <h4 className="text-white font-bold text-xl">Good Food</h4>
                  <p className="text-white text-sm">Lorem ipsum dolor sit amet consectetur adipisicing elit. Ab, quia.</p>
                </div>
              </div>

              {/* Service 2 */}
              <div className="col-md-6 d-flex flex-col items-stretch mt-4 md:mt-0">
                <div className="icon-circle pink hover:border-pink-600 hover:shadow-lg">
                  <div className="icon">
                    <i className="fas fa-umbrella-beach text-white text-3xl"></i>
                  </div>
                  <h4 className="text-white font-bold text-xl">Beach Resort</h4>
                  <p className="text-white text-sm">Lorem ipsum dolor sit amet consectetur adipisicing elit. Ab, quia.</p>
                </div>
              </div>

              {/* Service 3 */}
              <div className="col-md-6 d-flex flex-col items-stretch mt-4">
                <div className="icon-circle yellow hover:border-yellow-500 hover:shadow-lg">
                  <div className="icon">
                    <i className="fas fa-hiking text-white text-3xl"></i>
                  </div>
                  <h4 className="text-white font-bold text-xl">Mountain-climbing</h4>
                  <p className="text-white text-sm">Lorem ipsum dolor sit amet consectetur adipisicing elit. Ab, quia.</p>
                </div>
              </div>

              {/* Service 4 */}
              <div className="col-md-6 d-flex flex-col items-stretch mt-4">
                <div className="icon-circle blue hover:border-blue-900 hover:shadow-lg">
                  <div className="icon">
                    <i className="fas fa-mountain text-white text-3xl"></i>
                  </div>
                  <h4 className="text-white font-bold text-xl">Hot Air Balloon</h4>
                  <p className="text-white text-sm">Lorem ipsum dolor sit amet consectetur adipisicing elit. Ab, quia.</p>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
};

export default Services;
