import { Module } from '@nestjs/common';
import { ConfigModule } from '@nestjs/config';
import { SequelizeModule } from '@nestjs/sequelize';

// Feature Modules
import { SampleModule } from 'src/sample/sample.module';

// Sequelize Models
import { Sample } from 'src/sample/sample.model';
import { User } from 'src/users/users.model';
import { UserModule } from 'src/users/users.module';

@Module({
  imports: [
    ConfigModule.forRoot(),
    SequelizeModule.forRoot({
      dialect: 'mysql',
      host: process.env.SEQUELIZE_HOST,
      port: parseInt(process.env.SEQUELIZE_PORT),
      username: process.env.SEQUELIZE_USERNAME,
      password: process.env.SEQUELIZE_PASSWORD,
      database: process.env.SEQUELIZE_DATABASE,
      models: [Sample, User],
      autoLoadModels: true,
      synchronize: true,
    }),
    SampleModule,
    UserModule
  ],
  controllers: [],
  providers: [],
})
export class AppModule {}
